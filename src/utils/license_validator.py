# Fetch org and yash keys from env or Secrets Manager
# Expose for tests


# Dummy redis for test patching
import types
redis = types.SimpleNamespace(Redis=None)
"""
License validator for Data Insights A2A Server.

Startup flow:
  Redis cache hit → use cached APP_LICENSE
  Redis cache miss → fetch from AWS Secrets Manager → cache in Redis (TTL 1hr)
  → RSA-OAEP decrypt AES key → AES-256-CBC decrypt → RSA-PSS verify
  → validate expiry + agent name → inject PyArmor license
"""
import asyncio
import os
import json
import base64
import logging
import glob
from datetime import datetime, timezone, timedelta

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.utils.secret_manager import get_secret

logger = logging.getLogger(__name__)

SECRET_NAME = os.getenv("LICENSE_SECRET_NAME", "dev-pyarmor-license")
REDIS_LICENSE_KEY = "license:app_license"
REDIS_TTL = int(os.getenv("LICENSE_CACHE_TTL", 1))  # 1 hour default
LICENSE_RETRY_INTERVAL = int(os.getenv("LICENSE_RETRY_INTERVAL", 240000))

# Agent identity — hardcoded, never overridden by env
AGENT_SERVICE_ID   = "79d6c561-747c-4688-a110-5639b93a2459"  # shared across all data-insights-agent components
AGENT_COMPONENT_ID = "c983e1e4-2d20-4018-9a18-0f11fc36bf0d"  # unique to data-insights-backend-server
AGENT_COMPONENT_TYPE = "backend"


def get_agent_service_id() -> str:
    return AGENT_SERVICE_ID


def get_agent_component_id() -> str:
    return AGENT_COMPONENT_ID


_license_metadata: dict | None = None
_license_expires_at: datetime | None = None
_retry_task: asyncio.Task | None = None
_license_valid_event: asyncio.Event | None = None


def _get_sync_redis():
    """Create a sync Redis client using env vars."""
    import redis as sync_redis
    return sync_redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6380)),
        password=os.getenv("REDIS_PASSWORD", ""),
        ssl=os.getenv("REDIS_SSL", "true").lower() == "true",
        decode_responses=True,
    )


def _fetch_app_license() -> str:
    """Fetch APP_LICENSE from Redis cache, fallback to Secrets Manager."""

    sync_redis = _get_sync_redis()

    try:
        cached = sync_redis.get(REDIS_LICENSE_KEY)
        if cached:
            logger.info("License fetched from Redis cache", extra={"source": "redis", "ttl": REDIS_TTL})
            return cached
    except Exception as e:
        logger.warning(
            "Redis unavailable, falling back to Secrets Manager",
            extra={"error": str(e), "redis_key": REDIS_LICENSE_KEY},
            exc_info=True
        )

    try:
        logger.info("Fetching license from AWS Secrets Manager", extra={"secret_name": SECRET_NAME})
        secret = get_secret(SECRET_NAME)
        secret_data = json.loads(secret)
        app_license = secret_data.get("APP_LICENSE", secret_data.get("app_license", ""))

        if not app_license:
            logger.error(
                "APP_LICENSE not found in Secrets Manager secret",
                extra={"secret_name": SECRET_NAME, "keys_found": list(secret_data.keys())}
            )
            raise RuntimeError(f"APP_LICENSE not found in Secrets Manager secret '{SECRET_NAME}'")

        try:
            sync_redis.set(REDIS_LICENSE_KEY, app_license, ex=REDIS_TTL)
            logger.info(
                "License cached in Redis",
                extra={"ttl_seconds": REDIS_TTL, "redis_key": REDIS_LICENSE_KEY}
            )
        except Exception as e:
            logger.warning(
                "Failed to cache license in Redis",
                extra={"error": str(e), "redis_key": REDIS_LICENSE_KEY},
                exc_info=True
            )

        return app_license

    except Exception as e:
        logger.error(
            "Failed to fetch license from Secrets Manager",
            extra={"secret_name": SECRET_NAME, "error": str(e)},
            exc_info=True
        )
        raise RuntimeError(f"Failed to fetch license from Secrets Manager '{SECRET_NAME}': {e}")


def _calculate_expiry_date(metadata: dict) -> datetime:
    """Calculate license expiry date from metadata."""
    if "expires_at" in metadata:
        return datetime.fromisoformat(metadata["expires_at"]).replace(tzinfo=timezone.utc)
    
    if "tenure_days" in metadata:
        issued_at = datetime.fromisoformat(metadata["issued_at"]).replace(tzinfo=timezone.utc) if "issued_at" in metadata else datetime.now(timezone.utc)
        return issued_at + timedelta(days=metadata["tenure_days"])
    
    if "tenure_minutes" in metadata:
        issued_at = datetime.fromisoformat(metadata["issued_at"]).replace(tzinfo=timezone.utc) if "issued_at" in metadata else datetime.now(timezone.utc)
        return issued_at + timedelta(minutes=metadata["tenure_minutes"])
    
    raise RuntimeError("License has no expiry information (expires_at, tenure_days, or tenure_minutes)")


def _decrypt_license_data(license_package: dict, org_private_key_pem: str) -> tuple[bytes, dict]:
    """Decrypt and parse license data. Returns (decrypted_bytes, license_data)."""
    encrypted_aes_key = base64.b64decode(license_package["encrypted_key"])
    nonce = base64.b64decode(license_package["iv"])
    ciphertext = base64.b64decode(license_package["encrypted_data"])
    tag = base64.b64decode(license_package["tag"])

    try:
        org_private_key = serialization.load_pem_private_key(
            org_private_key_pem.encode(), password=None
        )
        aes_key = org_private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to decrypt AES key: {e}")

    try:
        aesgcm = AESGCM(aes_key)
        decrypted_bytes = aesgcm.decrypt(nonce, ciphertext + tag, None)
        license_data = json.loads(decrypted_bytes.decode())
    except Exception as e:
        raise RuntimeError(f"Failed to decrypt license data: {e}")
    
    return decrypted_bytes, license_data


def _verify_signature(decrypted_bytes: bytes, signature: bytes, yash_public_key_pem: str) -> None:
    """Verify license signature using public key."""
    try:
        yash_public_key = serialization.load_pem_public_key(
            yash_public_key_pem.encode()
        )
        yash_public_key.verify(
            signature,
            decrypted_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
    except Exception as e:
        raise RuntimeError(f"License signature verification failed: {e}")


def _validate_and_store(app_license: str) -> None:
    global _license_metadata, _license_expires_at

    org_private_key_pem = os.getenv("ORG_PRIVATE_KEY", "").strip().replace("\\n", "\n")
    yash_public_key_pem = os.getenv("YASH_PUBLIC_KEY", "").strip().replace("\\n", "\n")

    if not org_private_key_pem or not yash_public_key_pem:
        logger.info("ORG_PRIVATE_KEY/YASH_PUBLIC_KEY not in env, fetching from Secrets Manager",
                    extra={"secret_name": SECRET_NAME})
        try:
            secret = get_secret(SECRET_NAME)
            secret_data = json.loads(secret)
            org_private_key_pem = secret_data.get("ORG_PRIVATE_KEY", "").strip().replace("\\n", "\n")
            yash_public_key_pem = secret_data.get("YASH_PUBLIC_KEY", "").strip().replace("\\n", "\n")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch keys from Secrets Manager: {e}")

    if not org_private_key_pem or not yash_public_key_pem:
        raise RuntimeError("ORG_PRIVATE_KEY and YASH_PUBLIC_KEY not found in env or Secrets Manager")

    try:
        license_blob = base64.b64decode(app_license)
        license_package = json.loads(license_blob)
    except Exception as e:
        raise RuntimeError(f"Failed to decode APP_LICENSE: {e}")

    signature = base64.b64decode(license_package["signature"])
    decrypted_bytes, license_data = _decrypt_license_data(license_package, org_private_key_pem)
    _verify_signature(decrypted_bytes, signature, yash_public_key_pem)

    metadata = license_data.get("metadata", license_data)
    expires_at = _calculate_expiry_date(metadata)

    if datetime.now(timezone.utc) > expires_at:
        raise RuntimeError(f"License expired at {expires_at.isoformat()}")

    agent_name = os.getenv("AGENT_NAME", "").strip()
    licensed_agents = metadata.get("agents", [])
    if agent_name not in licensed_agents:
        raise RuntimeError(f"Agent '{agent_name}' is not licensed. Licensed agents: {licensed_agents}")

    pyarmor_license_hex = license_data.get("pyarmor_license", "")
    if pyarmor_license_hex:
        _inject_pyarmor_license(bytes.fromhex(pyarmor_license_hex))

    _license_metadata = metadata
    _license_expires_at = expires_at
    
    # Signal that license is now valid
    if _license_valid_event is not None:
        _license_valid_event.set()
    
    logger.info(f"✅ License validated. Agent: {agent_name}, Expires: {expires_at.isoformat()}")


def initialize_license() -> None:
    global _license_metadata
    if os.getenv("LICENSE_ENFORCE", "true").strip().lower() != "true":
        logger.info("License enforcement disabled (LICENSE_ENFORCE=false), skipping initialization")
        return
    if _license_metadata is not None:
        return
    try:
        app_license = _fetch_app_license()
        _validate_and_store(app_license)
    except RuntimeError as e:
        if "expired" in str(e).lower():
            logger.warning(
                "License expired at startup, app will retry in background",
                extra={"retry_interval_seconds": LICENSE_RETRY_INTERVAL, "error": str(e)}
            )
        else:
            logger.error("License initialization failed", extra={"error": str(e)}, exc_info=True)


async def _license_retry_loop() -> None:
    """Background task: retries license validation every LICENSE_RETRY_INTERVAL seconds."""
    global _retry_task
    
    if _license_valid_event is None:
        return
    
    while not _license_valid_event.is_set():
        logger.info(
            f"Retrying license validation in {LICENSE_RETRY_INTERVAL}s...",
            extra={"retry_interval_seconds": LICENSE_RETRY_INTERVAL}
        )
        
        try:
            # Wait for the event with timeout (returns True if set, False if timeout)
            await asyncio.wait_for(_license_valid_event.wait(), timeout=LICENSE_RETRY_INTERVAL)
        except asyncio.TimeoutError:
            # Timeout reached, try to refresh license
            try:
                refresh_license()
                logger.info("✅ License validated successfully on retry")
            except Exception as e:
                logger.warning("License retry failed", extra={"error": str(e)})
    
    _retry_task = None


def start_license_retry_task() -> None:
    """Called from app lifespan to start background retry if license was not valid at startup."""
    global _retry_task, _license_valid_event
    
    if _license_metadata is None and _retry_task is None:
        _license_valid_event = asyncio.Event()
        loop = asyncio.get_event_loop()
        _retry_task = loop.create_task(_license_retry_loop())
        logger.info(
            "License retry background task started",
            extra={"retry_interval_seconds": LICENSE_RETRY_INTERVAL}
        )


def refresh_license() -> None:
    """Force re-fetch from Secrets Manager (bypasses Redis cache) and re-validate."""
    global _license_metadata, _license_expires_at

    logger.info("Refreshing license from Secrets Manager", extra={"secret_name": SECRET_NAME})

    prev_metadata = _license_metadata
    prev_expires_at = _license_expires_at

    try:
        secret = get_secret(SECRET_NAME)
        secret_data = json.loads(secret)
        app_license = secret_data.get("APP_LICENSE", secret_data.get("app_license", ""))

        if not app_license:
            logger.error(
                "APP_LICENSE not found in Secrets Manager during refresh",
                extra={"secret_name": SECRET_NAME, "keys_found": list(secret_data.keys())}
            )
            raise RuntimeError(f"APP_LICENSE not found in Secrets Manager secret '{SECRET_NAME}'")

        sync_redis = _get_sync_redis()
        try:
            sync_redis.set(REDIS_LICENSE_KEY, app_license, ex=REDIS_TTL)
            logger.info(
                "License refreshed and cached in Redis",
                extra={"ttl_seconds": REDIS_TTL, "redis_key": REDIS_LICENSE_KEY}
            )
        except Exception as e:
            logger.warning(
                "Failed to cache refreshed license in Redis",
                extra={"error": str(e), "redis_key": REDIS_LICENSE_KEY},
                exc_info=True
            )

    except Exception as e:
        logger.error(
            "Failed to refresh license from Secrets Manager",
            extra={"secret_name": SECRET_NAME, "error": str(e)},
            exc_info=True
        )
        raise RuntimeError(f"Failed to refresh license from Secrets Manager '{SECRET_NAME}': {e}")

    _license_metadata = None
    _license_expires_at = None
    try:
        _validate_and_store(app_license)
    except Exception as e:
        _license_metadata = prev_metadata
        _license_expires_at = prev_expires_at
        raise


def _inject_pyarmor_license(license_bytes: bytes) -> None:
    patterns = [
        "/app/pyarmor_runtime_*/license.lic",
        "./pyarmor_runtime_*/license.lic",
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                with open(path, "wb") as f:
                    f.write(license_bytes)
                logger.info(f"✅ PyArmor license injected: {path}")
            except Exception as e:
                logger.warning(f"Failed to inject PyArmor license at {path}: {e}")


def get_license_metadata() -> dict | None:
    return _license_metadata


def get_license_expires_at() -> datetime | None:
    return _license_expires_at