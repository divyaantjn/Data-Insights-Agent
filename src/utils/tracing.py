"""
OpenTelemetry Tracing Configuration Module.
This module provides tracing setup for the Backend using OpenTelemetry.
It supports configurable OTLP endpoints via environment variables.
"""
import os
import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

logger = logging.getLogger(__name__)

# Store our TracerProvider for use by other modules (like phoenix_setup.py)
_tracer_provider = None


def get_tracer_provider():
    """Get the TracerProvider we created. Use this instead of trace.get_tracer_provider()
    to ensure you get OUR provider with the correct service.name."""
    global _tracer_provider
    if _tracer_provider is None:
        # Fall back to global if our setup hasn't run yet
        return trace.get_tracer_provider()
    return _tracer_provider


def setup_tracing(
    service_name: Optional[str] = None,
    service_environment: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    otlp_timeout: Optional[int] = None,
) -> bool:
    """
    Initialize OpenTelemetry tracing with configurable settings.
    
    Environment variables:
    - OTEL_SERVICE_NAME: Name of the service (default: "data-insights-backend")
    - OTEL_ENVIRONMENT: Deployment environment (default: "dev")
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (required)
    - OTEL_EXPORTER_OTLP_TIMEOUT: Export timeout in seconds (default: 10)
    - OTEL_TRACING_ENABLED: Enable/disable tracing (default: "true")
    """
    tracing_enabled = os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true"
    if not tracing_enabled:
        logger.info("OpenTelemetry tracing is disabled via OTEL_TRACING_ENABLED")
        return False
    
    _service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "data-insights-backend")
    _service_environment = service_environment or os.getenv("OTEL_ENVIRONMENT", "dev")
    _otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    _otlp_timeout = otlp_timeout or int(os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "10"))
    
    if not _otlp_endpoint:
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is not set. OpenTelemetry tracing is disabled. "
            "Set this environment variable to enable distributed tracing."
        )
        return False
    
    try:
        resource = Resource.create({
            "service.name": _service_name,
            "service.environment": _service_environment,
        })
        
        # Create OTLP exporter targeting your OTEL collector
        otlp_exporter = OTLPSpanExporter(
            endpoint=_otlp_endpoint,
            timeout=_otlp_timeout
        )
        
        # Use BatchSpanProcessor for production (async, batched export)
        span_processor = BatchSpanProcessor(otlp_exporter)
        
        # Create our TracerProvider with the correct resource (service.name)
        # We ALWAYS create our own, regardless of any existing provider
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(span_processor)
        
        # Set as the global provider - overrides any existing provider
        trace.set_tracer_provider(tracer_provider)
        
        # Store reference for use by phoenix_setup.py and other modules
        global _tracer_provider
        _tracer_provider = tracer_provider
        
        # CRITICAL: Instrument HTTP clients with our TracerProvider explicitly
        HTTPXClientInstrumentor().instrument(tracer_provider=tracer_provider)
        RequestsInstrumentor().instrument(tracer_provider=tracer_provider)
        
        logger.info(
            f"OpenTelemetry tracing initialized: service={_service_name}, "
            f"environment={_service_environment}, endpoint={_otlp_endpoint}"
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
        return False


def instrument_fastapi_app(app):
    """Instrument a FastAPI application for tracing."""
    try:
        # Use OUR TracerProvider, not the global one
        tracer_provider = get_tracer_provider()
        
        if tracer_provider and hasattr(tracer_provider, 'resource'):
            # Pass our tracer_provider explicitly to FastAPI instrumentor
            # This ensures the root HTTP span uses our provider with correct service.name
            FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
            
            resource = tracer_provider.resource
            service_name = resource.attributes.get("service.name", "unknown")
            logger.info(f"FastAPI application instrumented for tracing (service.name={service_name})")
        else:
            logger.debug("Tracing not initialized, skipping FastAPI instrumentation")
    except Exception as e:
        logger.warning(f"Failed to instrument FastAPI app: {e}")


def get_tracer(name: str = __name__):
    """Get a tracer instance for creating custom spans."""
    return trace.get_tracer(name)


def flush_traces(timeout_millis: int = 5000) -> bool:
    """
    Force flush all pending spans to the OTLP endpoint.
    
    CRITICAL FOR LAMBDA: BatchSpanProcessor queues spans in memory and exports
    them periodically. In Lambda, the function may freeze before spans are flushed.
    Call this function before returning from the Lambda handler to ensure traces
    are exported.
    
    Args:
        timeout_millis: Maximum time in milliseconds to wait for flush (default: 5000)
    
    Returns:
        True if flush was successful, False otherwise
    """
    try:
        # Use OUR TracerProvider to ensure we flush the right one
        tracer_provider = get_tracer_provider()
        
        if tracer_provider and hasattr(tracer_provider, 'force_flush'):
            success = tracer_provider.force_flush(timeout_millis)
            if success:
                logger.debug("Traces flushed successfully to OTLP endpoint")
            else:
                logger.warning("Trace flush timed out - some traces may not be exported")
            return success
        else:
            logger.warning("TracerProvider does not support force_flush")
            return False
            
    except Exception as e:
        logger.error(f"Failed to flush traces: {e}")
        return False
