"""
xray_tracing_middleware.py - Unified X-Ray and OpenTelemetry tracing middleware.

This middleware:
1. Creates X-Ray segments (or subsegments in Lambda) for each request
2. Sets annotations on X-Ray segments using safe_put_annotation()
3. Sets attributes on OTEL spans using safe_span_attr()
4. Captures user context from auth middleware
5. Captures HTTP metadata and errors

LAMBDA NOTE: In AWS Lambda, the Lambda runtime creates the parent X-Ray segment.
Therefore, this middleware creates SUBSEGMENTS instead of segments when running
in Lambda. This is detected via the AWS_LAMBDA_FUNCTION_NAME environment variable.
"""

import logging
import os
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core.exceptions.exceptions import AlreadyEndedException, SegmentNotFoundException
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode
xray_recorder.configure(context_missing='LOG_ERROR', sampling=False)
# Environment configuration
ENV = os.getenv("ENV", "DEV").lower()

logger = logging.getLogger(__name__)


class XRayTracingMiddleware(BaseHTTPMiddleware):
    """
    Unified X-Ray and OpenTelemetry tracing middleware.
    
    This middleware:
    1. Creates X-Ray segments (or subsegments in Lambda) for each request
    2. Sets annotations on X-Ray segments using safe_put_annotation()
    3. Sets attributes on OTEL spans using safe_span_attr()
    4. Captures user context from auth middleware
    5. Captures HTTP metadata and errors
    
    LAMBDA NOTE: In AWS Lambda, the Lambda runtime creates the parent X-Ray segment.
    Therefore, this middleware creates SUBSEGMENTS instead of segments when running
    in Lambda. This is detected via the AWS_LAMBDA_FUNCTION_NAME environment variable.
    
    CRITICAL: Must be added LAST (so it executes FIRST) to ensure 
    the segment exists when other middlewares run.
    """
    
    def __init__(self, app, service_name: str = "data-insights-backend"):
        super().__init__(app)
        self.service_name = service_name
        self.skip_paths = ["/health", "/health/redis", "/docs", "/openapi.json", "/redoc", "/", "/jobs"]
        logger.info(f"✅ XRayTracingMiddleware initialized: {service_name}")
    
    async def dispatch(self, request: Request, call_next):
        # Skip for health checks and docs
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        segment = None
        subsegment = None
        segment_ended = False
        
        # Detect if running in Lambda (Lambda sets AWS_LAMBDA_FUNCTION_NAME)
        is_lambda = os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None
        
        print("\n--- [XRAY TRACING DEBUG] ---")
        print(f"🔵 Starting trace for: {request.method} {request.url.path}")
        print(f"   Running in Lambda: {is_lambda}")
        
        try:
            # =====================================================
            # STEP 1: Begin X-Ray segment or subsegment
            # In Lambda, we MUST use subsegments because Lambda creates the parent segment
            # =====================================================
            if is_lambda:
                # In Lambda, use subsegment (Lambda already creates the parent segment)
                subsegment = xray_recorder.begin_subsegment(
                    name=f"{self.service_name}-{request.url.path}"
                )
                print(f"✅ X-Ray SUBsegment created: {subsegment.id if subsegment else 'None'}")
                logger.info(f"🔵 X-Ray subsegment created for {request.method} {request.url.path}")
            else:
                # Outside Lambda, create a full segment
                segment = xray_recorder.begin_segment(
                    name=self.service_name,
                    traceid=None,
                    parent_id=None,
                )
                print(f"✅ X-Ray segment created: {segment.id}")
                logger.info(f"🔵 X-Ray segment created: {segment.id} for {request.method} {request.url.path}")
            
            # Get the active entity (segment or subsegment)
            active_entity = subsegment if is_lambda else segment
            
            # Add HTTP request metadata
            if active_entity:
                try:
                    active_entity.put_http_meta('url', str(request.url))
                    active_entity.put_http_meta('method', request.method)
                except Exception as e:
                    logger.debug(f"Could not set HTTP meta: {e}")
            
            # =====================================================
            # STEP 2: Process the request (goes through other middlewares)
            # =====================================================
            response = await call_next(request)
            
            # =====================================================
            # STEP 3: Define helper functions for safe annotation/attribute setting
            # =====================================================
            status_code = getattr(response, 'status_code', 500)
            span = trace.get_current_span()
            
            # Debug: Check if we have an active span
            span_info = "NO SPAN" if not span else f"SPAN EXISTS (recording={span.is_recording()})"
            print(f"📊 Response status: {status_code}, OTEL Span: {span_info}")
            
            # Helper function for X-Ray SEGMENT/SUBSEGMENT annotations
            def safe_put_annotation(key: str, value) -> bool:
                """Safely set annotation on X-Ray segment or subsegment."""
                if value is None:
                    return False
                try:
                    if active_entity:
                        active_entity.put_annotation(key, value)
                        logger.debug(f"✅ X-Ray annotation: {key}={value}")
                        return True
                except (AlreadyEndedException, SegmentNotFoundException):
                    logger.debug(f"⚠️ Entity already ended, skipping: {key}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed X-Ray annotation {key}: {e}")
                return False
            
            # Helper function for OpenTelemetry SPAN attributes
            def safe_span_attr(key: str, value):
                """Safely set attribute on OpenTelemetry span."""
                if value is None:
                    return
                try:
                    if span and span.is_recording():
                        span.set_attribute(key, value)
                        logger.debug(f"✅ OTEL span attr: {key}={value}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed OTEL span attr {key}: {e}")
            
            # Add HTTP response metadata to segment/subsegment
            if active_entity:
                try:
                    active_entity.put_http_meta('status', status_code)
                except Exception as e:
                    logger.debug(f"Could not set HTTP response meta: {e}")
            
            # =====================================================
            # STEP 4: Set core HTTP annotations (segment + span)
            # =====================================================
            safe_put_annotation('http_status_code', status_code)
            safe_span_attr('http.status_code', status_code)
            
            safe_put_annotation('http_method', request.method)
            safe_span_attr('http.method', request.method)
            
            safe_put_annotation('request_success', status_code < 400)
            safe_span_attr('http.success', status_code < 400)
            
            safe_put_annotation('env', ENV)
            safe_span_attr('deployment.environment', ENV)
            
            # Set error/fault flags
            if status_code >= 400:
                safe_put_annotation('error', True)
                safe_span_attr('error', True)
            if status_code >= 500:
                safe_put_annotation('fault', True)
                safe_span_attr('http.fault', True)
                if span and span.is_recording():
                    span.set_status(Status(StatusCode.ERROR))
            
            # =====================================================
            # STEP 5: Capture user context from request.state
            # (set by auth middleware - request.state.user contains JWT payload)
            # =====================================================
            user_captured = False
            if hasattr(request.state, 'user') and request.state.user:
                user_payload = request.state.user
                user_captured = True
                
                # Extract user_id (sub claim in JWT)
                user_id = user_payload.get('sub')
                if user_id:
                    if safe_put_annotation('user_id', str(user_id)):
                        logger.debug(f"✅ Set user_id: {user_id}")
                    safe_span_attr('user.id', str(user_id))
                    print(f"👤 User ID captured: {user_id}")
                
                # Extract email (if present in JWT)
                user_email = user_payload.get('email') or user_payload.get('preferred_username')
                if user_email:
                    if safe_put_annotation('user_email', user_email):
                        logger.debug(f"✅ Set user_email: {user_email}")
                    safe_span_attr('user.email', user_email)
                
                # Extract username
                username = user_payload.get('preferred_username')
                if username:
                    safe_put_annotation('username', username)
                    safe_span_attr('user.name', username)
                
                # Extract realm from issuer
                issuer = user_payload.get('iss', '')
                if issuer:
                    realm = issuer.split('/')[-1] if '/' in issuer else issuer
                    safe_put_annotation('realm', realm)
                    safe_span_attr('auth.realm', realm)
                
                # Extract client_id
                client_id = user_payload.get('azp')
                if client_id:
                    safe_put_annotation('client_id', client_id)
                    safe_span_attr('auth.client_id', client_id)
                
                # Set auth mode
                safe_put_annotation('auth_mode', 'keycloak')
                safe_span_attr('auth.mode', 'keycloak')
            
            # Fallback: Check for individual state attributes
            if hasattr(request.state, 'user_id') and request.state.user_id:
                if safe_put_annotation('user_id', str(request.state.user_id)):
                    logger.debug(f"✅ Set user_id from state: {request.state.user_id}")
                safe_span_attr('user.id', str(request.state.user_id))
            
            if hasattr(request.state, 'user_email') and request.state.user_email:
                if safe_put_annotation('user_email', request.state.user_email):
                    logger.debug(f"✅ Set user_email from state: {request.state.user_email}")
                safe_span_attr('user.email', request.state.user_email)
            
            if not user_captured:
                print("⚠️ No user context found on request.state")
            
            # =====================================================
            # STEP 6: Capture error details if present
            # =====================================================
            if hasattr(request.state, 'error_details'):
                error_info = request.state.error_details
                safe_put_annotation('error_code', error_info.get('code'))
                safe_put_annotation('error_key', error_info.get('error_key'))
                safe_span_attr('error.code', error_info.get('code'))
                safe_span_attr('error.key', error_info.get('error_key'))
                logger.debug("✅ Set error annotations")
            
            # Capture agent info (for A2A scenarios)
            if hasattr(request.state, 'agent_id') and request.state.agent_id:
                safe_put_annotation('agent_id', str(request.state.agent_id))
                safe_span_attr('agent.id', str(request.state.agent_id))
            if hasattr(request.state, 'agent_name') and request.state.agent_name:
                safe_put_annotation('agent_name', request.state.agent_name)
                safe_span_attr('agent.name', request.state.agent_name)
            
            # Capture message_id (from request.state or X-Message-Id header)
            message_id = None
            if hasattr(request.state, 'message_id') and request.state.message_id:
                message_id = str(request.state.message_id)
            elif 'x-message-id' in request.headers:
                message_id = request.headers.get('x-message-id')
                request.state.message_id = message_id
            
            if message_id:
                safe_put_annotation('message_id', message_id)
                safe_span_attr('message_id', message_id)
                logger.info(f"✅ Set message_id trace: {message_id}")
            
            print(f"✅ Tracing complete for {request.url.path} [status={status_code}]")
            print("-----------------------------\n")
            logger.info(f"✅ Tracing complete for {request.url.path} [status={status_code}]")
            
            return response
            
        except Exception as e:
            # =====================================================
            # EXCEPTION HANDLING: Capture error in segment/subsegment + span
            # =====================================================
            if active_entity and not segment_ended:
                try:
                    active_entity.put_annotation('error', True)
                    active_entity.put_annotation('fault', True)
                    active_entity.put_annotation('error_type', type(e).__name__)
                    
                    import traceback
                    stack = traceback.extract_tb(e.__traceback__)
                    active_entity.add_exception(e, stack)
                    
                    logger.error(f"❌ Exception captured in X-Ray: {type(e).__name__}")
                except Exception as seg_error:
                    logger.warning(f"Could not add exception to X-Ray entity: {seg_error}")
            
            # Also record in OTEL span
            span = trace.get_current_span()
            if span and span.is_recording():
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", type(e).__name__)
            
            raise
            
        finally:
            # =====================================================
            # CLEANUP: End the X-Ray segment or subsegment
            # =====================================================
            if not segment_ended:
                try:
                    if is_lambda and subsegment:
                        # In Lambda, end the subsegment
                        xray_recorder.end_subsegment()
                        logger.debug(f"🔵 X-Ray subsegment ended for {request.url.path}")
                    elif segment:
                        # Outside Lambda, end the full segment
                        xray_recorder.end_segment()
                        logger.debug(f"🔵 X-Ray segment ended for {request.url.path}")
                    segment_ended = True
                except Exception as e:
                    logger.warning(f"Failed to end X-Ray entity: {e}")
