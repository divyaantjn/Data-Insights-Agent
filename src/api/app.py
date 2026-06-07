from fastapi import FastAPI
from src.utils.config_loader import ConfigLoader
from contextlib import asynccontextmanager
from middleware.auth_middleware import KeycloakAuthMiddleware
from src.utils.opik_setup import setup_opik_tracing, flush_traces, OPIKMiddleware
import logging
import os
import asyncio

logger = logging.getLogger(__name__)

setup_opik_tracing(
    llm_only=False,
    enable_litellm=True,
    enable_genai=False
)

from src.api.routes import router, initialize_connection_pool, initialize_database, close_connection_pool

# Import heartbeat client and start function
from src.utils.heartbeat import heartbeat_client, start_periodic_heartbeat

# Global flag to control heartbeat task
_heartbeat_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context - runs at startup and shutdown."""
    global _heartbeat_task
    
    try:
        initialize_connection_pool()
        initialize_database()
        logger.info("✓ Application startup complete")
        
        # Start periodic heartbeat task (only for EKS container, not Lambda)
        if os.getenv("AWS_LAMBDA_FUNCTION_NAME") is None:
            _heartbeat_task = await start_periodic_heartbeat(interval=30)
    except Exception as e:
        logger.error(f"✗ Application startup failed: {e}", exc_info=True)
        raise

    yield  # App is running

    # Stop heartbeat task
    if _heartbeat_task and not _heartbeat_task.done():
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
            raise
    
    flush_traces(timeout=30)
    close_connection_pool()

def create_app() -> FastAPI:
    config = ConfigLoader()
    app_config = config.get_app_config()
    
    app = FastAPI(
        title=app_config['name'],
        version=app_config['version'],
        debug=app_config['debug'],
        lifespan=lifespan
    )   


    app.add_middleware(OPIKMiddleware)

    app.add_middleware(KeycloakAuthMiddleware)
    
    app.include_router(router)
    
    return app