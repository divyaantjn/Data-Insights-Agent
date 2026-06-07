import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse

from src.api.app import create_app
from src.utils.config_loader import ConfigLoader
from middleware.auth_middleware import KeycloakAuthMiddleware
from middleware.license_middleware import LicenseMiddleware

# Import error capture utilities
from src.utils.error_capture import (
    capture_http_exception,
    capture_validation_error,
    capture_exception
)

# Initialize OpenTelemetry tracing FIRST (before phoenix/opik)
from src.utils.tracing import setup_tracing, instrument_fastapi_app
setup_tracing()

config = ConfigLoader()
app_config = config.get_app_config()
app = create_app()   # 👉 Make it global

# ============================================================
# EXCEPTION HANDLERS - for consistent error capture and tracing
# ============================================================
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with span attributes for tracing."""
    capture_http_exception(exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with span attributes for tracing."""
    capture_validation_error(exc)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with span attributes for tracing."""
    capture_exception(exc, {"request_path": str(request.url.path)})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# ============================================================
# MIDDLEWARE ORDER: Auth first, then XRay last (closest to request)
# ============================================================
app.add_middleware(KeycloakAuthMiddleware)
app.add_middleware(LicenseMiddleware)
# # Import and add XRayTracingMiddleware LAST (runs first on requests)
from middleware.xray_tracing_middleware import XRayTracingMiddleware
app.add_middleware(XRayTracingMiddleware)

# Instrument FastAPI app for OpenTelemetry tracing
instrument_fastapi_app(app)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=app_config["host"],
        port=app_config["port"]
    )
