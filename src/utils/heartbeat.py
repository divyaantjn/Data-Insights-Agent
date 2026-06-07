import os
import httpx
import requests
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from src.utils.license_validator import AGENT_SERVICE_ID, AGENT_COMPONENT_ID, AGENT_COMPONENT_TYPE

logger = logging.getLogger(__name__)

class HeartbeatClient:
    """Client to send heartbeat signals to license backend"""

    def __init__(self):
        self.license_backend_url = os.getenv(
            "LICENSE_BACKEND_URL",
            ""
        )
        self.component_id = AGENT_COMPONENT_ID
        self.agent_name = os.getenv("SERVER_NAME", "")
        self.service_id = AGENT_SERVICE_ID
        self.component_type = AGENT_COMPONENT_TYPE
        self.heartbeat_endpoint = f"{self.license_backend_url}/api/heartbeat"
    
    def _build_metadata(self, metadata: Optional[Dict[str, Any]] = None, heartbeat_type: str = "periodic", api_path: Optional[str] = None) -> Dict[str, Any]:
        """Build metadata with timestamp, heartbeat type, and API path"""
        base_metadata = {
            "heartbeat_type": heartbeat_type,
            "component_id": self.component_id,
            "component_type": self.component_type,
            "service_id": self.service_id,
            "agent_name": self.agent_name,
            "environment": os.getenv("ENV", "dev")
        }
        app_version = os.getenv("APP_VERSION")
        if app_version and app_version != "unknown":
            base_metadata["version"] = app_version
        if api_path:
            base_metadata["api_path"] = api_path
            base_metadata["execution_context"] = "api_request"
        else:
            base_metadata["execution_context"] = "background_task"
        if metadata:
            base_metadata.update(metadata)
        return base_metadata
    
    async def send_heartbeat(self, status: str = "healthy", metadata: Optional[Dict[str, Any]] = None, heartbeat_type: str = "periodic", api_path: Optional[str] = None) -> bool:
        """Send heartbeat to license backend"""
        try:
            full_metadata = self._build_metadata(metadata, heartbeat_type, api_path)
            payload = {
                "component_id": self.component_id,
                "agent_name": self.agent_name,
                "service_id": self.service_id,
                "status": status,
                "metadata": full_metadata
            }
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(self.heartbeat_endpoint, json=payload)
                
                if response.status_code == 200:
                    logger.info(f"✅ Heartbeat sent ({heartbeat_type}): {self.agent_name}")
                    return True
                else:
                    logger.warning(f"⚠️ Heartbeat failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Heartbeat error: {str(e)}")
            return False
    
    async def send_execution_heartbeat(self, api_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send heartbeat on API request execution"""
        exec_metadata = {"triggered_by": "api_request"}
        if metadata:
            exec_metadata.update(metadata)
        result = await self.send_heartbeat(status="healthy", metadata=exec_metadata, heartbeat_type="execution", api_path=api_path)
        if result and metadata and "request_id" in metadata:
            logger.info(f"  Request ID: {metadata['request_id']}")
        return result
    
    async def send_completion_heartbeat(self, api_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send heartbeat when API execution completes (immediately after execution)"""
        completion_metadata = {"triggered_by": "api_completion"}
        if metadata:
            completion_metadata.update(metadata)
        return await self.send_heartbeat(status="healthy", metadata=completion_metadata, heartbeat_type="completion", api_path=api_path)
    
    async def send_periodic_heartbeat(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send periodic heartbeat update (background task)"""
        periodic_metadata = {"triggered_by": "periodic_scheduler"}
        if metadata:
            periodic_metadata.update(metadata)
        return await self.send_heartbeat(status="healthy", metadata=periodic_metadata, heartbeat_type="periodic", api_path=None)

# Global instance
heartbeat_client = HeartbeatClient()

async def start_periodic_heartbeat(interval: int = 30):
    """Start periodic heartbeat task for EKS/long-running processes
    
    Args:
        interval: Heartbeat interval in seconds (default: 30)
        
    Returns:
        asyncio.Task: The heartbeat task that can be cancelled on shutdown
    """
    import asyncio
    
    async def _periodic_heartbeat():
        """Send heartbeat every interval seconds"""
        try:
            while True:
                try:
                    await asyncio.sleep(interval)
                    await heartbeat_client.send_periodic_heartbeat()
                except asyncio.CancelledError:
                    logger.info("❌ Heartbeat task cancelled")
                    raise
                except Exception as e:
                    logger.error(f"❌ Periodic heartbeat error: {str(e)}")
        except asyncio.CancelledError:
            logger.info("✓ Heartbeat task stopped")
    
    task = asyncio.create_task(_periodic_heartbeat())
    logger.info(f"✓ Periodic heartbeat started ({interval}s interval)")
    return task

def send_heartbeat_sync(status: str = "healthy", metadata: Optional[Dict[str, Any]] = None, heartbeat_type: str = "periodic", api_path: Optional[str] = None) -> bool:
    """Synchronous heartbeat for Lambda - uses requests instead of httpx"""
    try:
        license_backend_url = os.getenv(
            "LICENSE_BACKEND_URL",
            ""
        )
        agent_name = os.getenv("SERVER_NAME", "")

        full_metadata = {
            "heartbeat_type": heartbeat_type,
            "component_id": AGENT_COMPONENT_ID,
            "component_type": AGENT_COMPONENT_TYPE,
            "service_id": AGENT_SERVICE_ID,
            "agent_name": agent_name,
            "environment": os.getenv("ENV", "dev")
        }
        app_version = os.getenv("APP_VERSION")
        if app_version and app_version != "unknown":
            full_metadata["version"] = app_version
        if api_path:
            full_metadata["api_path"] = api_path
            full_metadata["execution_context"] = "api_request"
        else:
            full_metadata["execution_context"] = "background_task"
        if metadata:
            full_metadata.update(metadata)

        payload = {
            "component_id": AGENT_COMPONENT_ID,
            "agent_name": agent_name,
            "service_id": AGENT_SERVICE_ID,
            "status": status,
            "metadata": full_metadata
        }

        response = requests.post(
            f"{license_backend_url}/api/heartbeat",
            json=payload,
            timeout=5.0
        )
        
        if response.status_code == 200:
            logger.info(f"✅ Heartbeat sent ({heartbeat_type}): {agent_name}")
            return True
        else:
            logger.warning(f"⚠️ Heartbeat failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Heartbeat error: {str(e)}")
        return False

# def send_completion_heartbeat_sync(api_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
#     """Synchronous completion heartbeat for Lambda (API execution complete)"""
#     completion_metadata = {"triggered_by": "api_completion"}
#     if metadata:
#         completion_metadata.update(metadata)
#     return send_heartbeat_sync(status="healthy", metadata=completion_metadata, heartbeat_type="completion", api_path=api_path)

def send_periodic_heartbeat_sync(metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Synchronous periodic heartbeat for Lambda (background task)"""
    periodic_metadata = {"triggered_by": "periodic_scheduler"}
    if metadata:
        periodic_metadata.update(metadata)
    return send_heartbeat_sync(status="healthy", metadata=periodic_metadata, heartbeat_type="periodic", api_path=None)
