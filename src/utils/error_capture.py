"""
error_capture.py - Error capture utilities for X-Ray and OpenTelemetry tracing.

This module provides utilities for capturing HTTP errors, validation errors,
and external API errors into X-Ray annotations and OTEL span attributes.
"""

import logging
import traceback
from typing import Dict, Any, List, Optional
from src.utils.otel_utils import set_xray_annotation
from opentelemetry import trace

logger = logging.getLogger(__name__)


def capture_http_exception(exc):
    """
    Capture an HTTP exception (HTTPException from FastAPI/Starlette).
    
    Args:
        exc: An HTTPException object with status_code and detail attributes
    """
    try:
        status_code = getattr(exc, 'status_code', 500)
        detail = getattr(exc, 'detail', str(exc))
        
        set_xray_annotation("error_occurred", True)
        set_xray_annotation("error_status_code", status_code)
        set_xray_annotation("error_message", str(detail)[:200])
        
        # Categorize the error
        if status_code >= 500:
            error_category = "server_error"
            error_severity = "critical"
        elif status_code >= 400:
            error_category = "client_error"
            error_severity = "warning"
        else:
            error_category = "unknown"
            error_severity = "info"
        
        set_xray_annotation("error_category", error_category)
        set_xray_annotation("error_severity", error_severity)
        
        # Also set on OTEL span
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("error", True)
            span.set_attribute("error.status_code", status_code)
            span.set_attribute("error.category", error_category)
            span.set_attribute("error.severity", error_severity)
            span.set_attribute("error.message", str(detail)[:500])
        
        logger.debug(f"✅ HTTP exception captured: status={status_code}, detail={detail}")
        
    except Exception as e:
        logger.warning(f"Failed to capture HTTP exception: {e}")


def capture_validation_error(exc):
    """
    Capture a validation error exception from FastAPI/Pydantic.
    
    Args:
        exc: A RequestValidationError exception object
    """
    try:
        # Get errors list from exception
        errors = []
        if hasattr(exc, 'errors') and callable(exc.errors):
            errors = exc.errors()
        elif hasattr(exc, 'errors'):
            errors = exc.errors
        elif hasattr(exc, 'args') and exc.args:
            errors = [{"msg": str(exc.args[0])}]
        
        set_xray_annotation("error_occurred", True)
        set_xray_annotation("error_type", "validation_error")
        set_xray_annotation("error_status_code", 422)
        set_xray_annotation("error_count", len(errors))
        
        # Capture first error details
        if errors and len(errors) > 0:
            first_error = errors[0]
            
            # Extract field location
            if isinstance(first_error, dict):
                if "loc" in first_error:
                    loc = first_error["loc"]
                    field = loc[-1] if loc else "unknown"
                    set_xray_annotation("error_field", str(field))
                    set_xray_annotation("error_location", ".".join(str(l) for l in loc))
                
                # Extract error type and message
                if "type" in first_error:
                    set_xray_annotation("validation_error_type", first_error["type"])
                if "msg" in first_error:
                    set_xray_annotation("error_message", str(first_error["msg"])[:200])
        
        # Also set on OTEL span
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("error", True)
            span.set_attribute("error.type", "validation_error")
            span.set_attribute("error.count", len(errors))
        
        logger.debug(f"✅ Validation error captured: {len(errors)} errors")
        
    except Exception as e:
        logger.warning(f"Failed to capture validation error: {e}")


def capture_exception(exc, context: Optional[Dict[str, Any]] = None):
    """
    Capture a general exception with optional context.
    
    Args:
        exc: Any exception object
        context: Optional dict with additional context info
    """
    try:
        exc_type = type(exc).__name__
        exc_message = str(exc)[:500]
        exc_traceback = traceback.format_exc()[:1000]
        
        set_xray_annotation("error_occurred", True)
        set_xray_annotation("error_type", exc_type)
        set_xray_annotation("error_message", exc_message[:200])
        set_xray_annotation("error_category", "unhandled_exception")
        set_xray_annotation("error_severity", "critical")
        
        # Add context if provided
        if context:
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    set_xray_annotation(f"error_context_{key}", str(value)[:200])
        
        # Also set on OTEL span
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("error", True)
            span.set_attribute("error.type", exc_type)
            span.set_attribute("error.message", exc_message)
            span.set_attribute("exception.stacktrace", exc_traceback)
            span.record_exception(exc)
            
            if context:
                for key, value in context.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(f"error.context.{key}", str(value))
        
        logger.debug(f"✅ Exception captured: type={exc_type}, message={exc_message[:100]}")
        
    except Exception as e:
        logger.warning(f"Failed to capture exception: {e}")


def capture_http_error_details(status_code: int, error_details: Dict[str, Any] = None):
    """
    Capture HTTP error details as X-Ray annotations and span attributes.
    
    Args:
        status_code: The HTTP status code
        error_details: Optional dict with error info (message, error_key, details, etc.)
    """
    try:
        # Set basic error info
        set_xray_annotation("error_occurred", True)
        set_xray_annotation("error_status_code", status_code)
        
        # Categorize the error
        if status_code >= 500:
            error_category = "server_error"
            error_severity = "critical"
        elif status_code >= 400:
            error_category = "client_error"
            error_severity = "warning"
        else:
            error_category = "unknown"
            error_severity = "info"
        
        set_xray_annotation("error_category", error_category)
        set_xray_annotation("error_severity", error_severity)
        
        # Extract specific error details
        if error_details:
            if "message" in error_details:
                # Truncate message to 200 chars for annotation
                set_xray_annotation("error_message", str(error_details["message"])[:200])
            
            if "error_key" in error_details:
                set_xray_annotation("error_key", error_details["error_key"])
            
            if "details" in error_details:
                # For validation-like errors, capture the field
                details = error_details["details"]
                if isinstance(details, list) and len(details) > 0:
                    first_detail = details[0]
                    if isinstance(first_detail, dict):
                        if "loc" in first_detail:
                            loc = first_detail["loc"]
                            field = loc[-1] if loc else "unknown"
                            set_xray_annotation("error_field", str(field))
                        if "type" in first_detail:
                            set_xray_annotation("error_type", first_detail["type"])
        
        # Also set on OTEL span
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("error", True)
            span.set_attribute("error.status_code", status_code)
            span.set_attribute("error.category", error_category)
            span.set_attribute("error.severity", error_severity)
            if error_details and "message" in error_details:
                span.set_attribute("error.message", str(error_details["message"])[:500])
        
        logger.debug(f"✅ HTTP error captured: status={status_code}, category={error_category}")
        
    except Exception as e:
        logger.warning(f"Failed to capture HTTP error details: {e}")


def capture_validation_error_list(errors: List[Dict[str, Any]]):
    """
    Capture validation errors from a list of error dicts.
    
    Args:
        errors: List of validation error dicts from RequestValidationError.errors()
    
    Note: Use capture_validation_error(exc) for exception objects instead.
    """
    try:
        set_xray_annotation("error_occurred", True)
        set_xray_annotation("error_type", "validation_error")
        set_xray_annotation("error_status_code", 422)
        set_xray_annotation("error_count", len(errors))
        
        # Capture first error details
        if errors and len(errors) > 0:
            first_error = errors[0]
            
            # Extract field location
            if "loc" in first_error:
                loc = first_error["loc"]
                field = loc[-1] if loc else "unknown"
                set_xray_annotation("error_field", str(field))
                set_xray_annotation("error_location", ".".join(str(l) for l in loc))
            
            # Extract error type and message
            if "type" in first_error:
                set_xray_annotation("validation_error_type", first_error["type"])
            if "msg" in first_error:
                set_xray_annotation("error_message", str(first_error["msg"])[:200])
        
        # Also set on OTEL span
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("error", True)
            span.set_attribute("error.type", "validation_error")
            span.set_attribute("error.count", len(errors))
        
        logger.debug(f"✅ Validation error captured: {len(errors)} errors")
        
    except Exception as e:
        logger.warning(f"Failed to capture validation error: {e}")


def capture_external_api_error(
    service_name: str,
    endpoint: str,
    status_code: int = None,
    error_message: str = None,
    error_type: str = None
):
    """
    Capture errors from external API calls (e.g., to other microservices).
    
    Args:
        service_name: Name of the external service
        endpoint: The API endpoint called
        status_code: HTTP status code (if available)
        error_message: Error message
        error_type: Type of error (e.g., 'timeout', 'connection_error')
    """
    try:
        set_xray_annotation("external_api_error", True)
        set_xray_annotation("external_service", service_name)
        set_xray_annotation("external_endpoint", endpoint[:200])
        
        if status_code:
            set_xray_annotation("external_status_code", status_code)
        if error_type:
            set_xray_annotation("external_error_type", error_type)
        if error_message:
            set_xray_annotation("external_error_message", str(error_message)[:200])
        
        # Also set on OTEL span
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("external.api.error", True)
            span.set_attribute("external.service.name", service_name)
            span.set_attribute("external.endpoint", endpoint[:500])
            if status_code:
                span.set_attribute("external.status_code", status_code)
            if error_type:
                span.set_attribute("external.error.type", error_type)
        
        logger.debug(f"✅ External API error captured: service={service_name}, endpoint={endpoint}")
        
    except Exception as e:
        logger.warning(f"Failed to capture external API error: {e}")
