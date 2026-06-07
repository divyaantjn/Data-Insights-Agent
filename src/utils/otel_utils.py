"""
otel_utils.py - X-Ray annotation utilities for OpenTelemetry tracing.

This module provides utilities for setting X-Ray annotations and metadata
that are searchable in AWS X-Ray console and Grafana.
"""

import logging
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core.exceptions.exceptions import AlreadyEndedException, SegmentNotFoundException
from opentelemetry import trace

logger = logging.getLogger(__name__)


def set_xray_annotation(key: str, value):
    """
    Set an annotation on the current X-Ray segment.
    
    Annotations are indexed and searchable in X-Ray console.
    Only string, number, and boolean types are supported.
    
    Args:
        key: The annotation key (will have '.' replaced with '_')
        value: The annotation value
    """
    if value is None:
        return
    
    try:
        # Convert value to X-Ray compatible type
        if isinstance(value, bool):
            xray_value = value
        elif isinstance(value, (int, float)):
            xray_value = value
        elif isinstance(value, str):
            # X-Ray annotation values are limited to 500 chars
            xray_value = value[:500]
        else:
            xray_value = str(value)[:500]
        
        entity = xray_recorder.current_segment()
        if entity:
            # X-Ray annotation keys cannot contain '.'
            safe_key = key.replace('.', '_')[:500]
            entity.put_annotation(safe_key, xray_value)
            logger.debug(f"✅ X-Ray ANNOTATION set: {safe_key}={xray_value}")
        else:
            logger.warning(f"❌ No X-Ray segment for annotation: {key}")
    except (AlreadyEndedException, SegmentNotFoundException):
        logger.debug(f"⚠️ X-Ray segment already ended, skipping: {key}")
    except Exception as e:
        logger.warning(f"❌ Failed X-Ray annotation {key}={value}: {e}")


def set_xray_metadata(namespace: str, key: str, value):
    """
    Set metadata on the current X-Ray segment.
    
    Metadata is NOT indexed/searchable but can store larger/complex data.
    
    Args:
        namespace: The metadata namespace
        key: The metadata key
        value: The metadata value (can be any JSON-serializable type)
    """
    if value is None:
        return
    
    try:
        entity = xray_recorder.current_segment()
        if entity:
            entity.put_metadata(key, value, namespace)
            logger.debug(f"✅ X-Ray METADATA set: {namespace}/{key}")
        else:
            logger.warning(f"❌ No X-Ray segment for metadata: {namespace}/{key}")
    except (AlreadyEndedException, SegmentNotFoundException):
        logger.debug(f"⚠️ X-Ray segment already ended, skipping metadata: {key}")
    except Exception as e:
        logger.warning(f"❌ Failed X-Ray metadata {namespace}/{key}: {e}")


def force_user_context_to_xray(user_id: str = None, user_email: str = None, auth_mode: str = None):
    """
    Force user context annotations onto the current X-Ray segment.
    
    Use this when you need to set user context outside of middleware,
    for example in route handlers or service functions.
    
    Args:
        user_id: The user's ID
        user_email: The user's email
        auth_mode: The authentication mode (e.g., 'keycloak', 'api_key')
    """
    if user_id:
        set_xray_annotation('user_id', str(user_id))
    if user_email:
        set_xray_annotation('user_email', user_email)
    if auth_mode:
        set_xray_annotation('auth_mode', auth_mode)
    
    # Also set on OTEL span
    span = trace.get_current_span()
    if span and span.is_recording():
        if user_id:
            span.set_attribute('user.id', str(user_id))
        if user_email:
            span.set_attribute('user.email', user_email)
        if auth_mode:
            span.set_attribute('auth.mode', auth_mode)
