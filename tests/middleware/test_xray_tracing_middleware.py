"""
test_xray_tracing_middleware.py - Comprehensive unit tests for XRayTracingMiddleware.

Coverage targets:
- dispatch() - all branches (Lambda vs non-Lambda, skip paths, segment/subsegment creation)
- safe_put_annotation() - success, None value, AlreadyEndedException, SegmentNotFoundException, generic Exception
- safe_span_attr() - success, None value, generic Exception
- User context capture (request.state.user, fallback state attrs)
- HTTP metadata and error annotations
- Exception handling block
- Finally/cleanup block (Lambda subsegment, non-Lambda segment, already-ended)
- Agent / message_id capture
"""

import os
import traceback
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Lightweight stubs so we can import the module without real AWS / OTEL libs
# ---------------------------------------------------------------------------

import sys
import types

# ── aws_xray_sdk stubs ──────────────────────────────────────────────────────
xray_mod = types.ModuleType("aws_xray_sdk")
xray_core = types.ModuleType("aws_xray_sdk.core")
xray_exc_pkg = types.ModuleType("aws_xray_sdk.core.exceptions")
xray_exc = types.ModuleType("aws_xray_sdk.core.exceptions.exceptions")


class AlreadyEndedException(Exception):
    pass


class SegmentNotFoundException(Exception):
    pass


xray_exc.AlreadyEndedException = AlreadyEndedException
xray_exc.SegmentNotFoundException = SegmentNotFoundException

mock_recorder = MagicMock()
xray_core.xray_recorder = mock_recorder

sys.modules["aws_xray_sdk"] = xray_mod
sys.modules["aws_xray_sdk.core"] = xray_core
sys.modules["aws_xray_sdk.core.exceptions"] = xray_exc_pkg
sys.modules["aws_xray_sdk.core.exceptions.exceptions"] = xray_exc

# ── opentelemetry stubs ─────────────────────────────────────────────────────
otel_mod = types.ModuleType("opentelemetry")
otel_trace = types.ModuleType("opentelemetry.trace")
otel_status = types.ModuleType("opentelemetry.trace.status")


class StatusCode:
    ERROR = "ERROR"
    OK = "OK"


class Status:
    def __init__(self, code):
        self.code = code


otel_status.Status = Status
otel_status.StatusCode = StatusCode

mock_otel_span = MagicMock()
otel_trace.get_current_span = MagicMock(return_value=mock_otel_span)

sys.modules["opentelemetry"] = otel_mod
sys.modules["opentelemetry.trace"] = otel_trace
sys.modules["opentelemetry.trace.status"] = otel_status

# ── fastapi / starlette stubs ───────────────────────────────────────────────
fastapi_mod = types.ModuleType("fastapi")


class Request:
    pass


fastapi_mod.Request = Request
sys.modules["fastapi"] = fastapi_mod

starlette_mod = types.ModuleType("starlette")
starlette_mid = types.ModuleType("starlette.middleware")
starlette_base = types.ModuleType("starlette.middleware.base")


class BaseHTTPMiddleware:
    def __init__(self, app, **kwargs):
        self.app = app


starlette_base.BaseHTTPMiddleware = BaseHTTPMiddleware
sys.modules["starlette"] = starlette_mod
sys.modules["starlette.middleware"] = starlette_mid
sys.modules["starlette.middleware.base"] = starlette_base

# ---------------------------------------------------------------------------
# Now import the module under test
# ---------------------------------------------------------------------------
from middleware.xray_tracing_middleware import XRayTracingMiddleware  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_request(path="/api/data", method="GET", headers=None, state_attrs=None):
    """Build a minimal mock Request."""
    req = MagicMock()
    req.url.path = path
    req.method = method
    req.headers = headers or {}
    # Give request.state a plain object so hasattr works correctly
    state = MagicMock(spec=[])  # spec=[] means no attrs by default
    if state_attrs:
        for k, v in state_attrs.items():
            setattr(state, k, v)
    req.state = state
    return req


def make_response(status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    return resp


def make_middleware():
    app = MagicMock()
    return XRayTracingMiddleware(app, service_name="test-service")


# ---------------------------------------------------------------------------
# Fixtures that reset shared mocks between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mocks():
    mock_recorder.reset_mock()
    mock_otel_span.reset_mock()
    mock_otel_span.is_recording.return_value = True
    otel_trace.get_current_span.return_value = mock_otel_span
    yield


# ===========================================================================
# 1. Initialisation
# ===========================================================================

class TestInit:
    def test_service_name_stored(self):
        mw = make_middleware()
        assert mw.service_name == "test-service"

    def test_default_service_name(self):
        mw = XRayTracingMiddleware(MagicMock())
        assert mw.service_name == "data-insights-backend"

    def test_skip_paths_contains_health(self):
        mw = make_middleware()
        assert "/health" in mw.skip_paths

    def test_skip_paths_contains_docs(self):
        mw = make_middleware()
        assert "/docs" in mw.skip_paths


# ===========================================================================
# 2. Skip paths
# ===========================================================================

class TestSkipPaths:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["/health", "/health/redis", "/docs",
                                       "/openapi.json", "/redoc", "/", "/jobs"])
    async def test_skip_path_calls_next_directly(self, path):
        mw = make_middleware()
        req = make_request(path=path)
        expected_resp = make_response()
        call_next = AsyncMock(return_value=expected_resp)

        resp = await mw.dispatch(req, call_next)

        call_next.assert_awaited_once_with(req)
        assert resp is expected_resp
        mock_recorder.begin_segment.assert_not_called()
        mock_recorder.begin_subsegment.assert_not_called()


# ===========================================================================
# 3. Non-Lambda path – segment creation & cleanup
# ===========================================================================

class TestNonLambdaSegment:
    @pytest.mark.asyncio
    async def test_begin_segment_called(self):
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            # Ensure AWS_LAMBDA_FUNCTION_NAME is absent
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)

        mock_recorder.begin_segment.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_segment_called_in_finally(self):
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)

        mock_recorder.end_segment.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_meta_set_on_segment(self):
        mw = make_middleware()
        req = make_request(path="/api/test", method="POST")
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(201))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)

        mock_segment.put_http_meta.assert_any_call("method", "POST")

    @pytest.mark.asyncio
    async def test_http_meta_status_set(self):
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)

        mock_segment.put_http_meta.assert_any_call("status", 200)


# ===========================================================================
# 4. Lambda path – subsegment creation & cleanup
# ===========================================================================

class TestLambdaSubsegment:
    @pytest.mark.asyncio
    async def test_begin_subsegment_called_in_lambda(self):
        mw = make_middleware()
        req = make_request()
        mock_subseg = MagicMock()
        mock_recorder.begin_subsegment.return_value = mock_subseg
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {"AWS_LAMBDA_FUNCTION_NAME": "my-func"}):
            await mw.dispatch(req, call_next)

        mock_recorder.begin_subsegment.assert_called_once()
        mock_recorder.begin_segment.assert_not_called()

    @pytest.mark.asyncio
    async def test_end_subsegment_called_in_lambda(self):
        mw = make_middleware()
        req = make_request()
        mock_subseg = MagicMock()
        mock_recorder.begin_subsegment.return_value = mock_subseg
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {"AWS_LAMBDA_FUNCTION_NAME": "my-func"}):
            await mw.dispatch(req, call_next)

        mock_recorder.end_subsegment.assert_called_once()
        mock_recorder.end_segment.assert_not_called()

    @pytest.mark.asyncio
    async def test_subsegment_name_contains_path(self):
        mw = make_middleware()
        req = make_request(path="/api/charts")
        mock_subseg = MagicMock()
        mock_recorder.begin_subsegment.return_value = mock_subseg
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {"AWS_LAMBDA_FUNCTION_NAME": "fn"}):
            await mw.dispatch(req, call_next)

        call_args = mock_recorder.begin_subsegment.call_args
        assert "/api/charts" in call_args[1]["name"] or "/api/charts" in str(call_args)


# ===========================================================================
# 5. HTTP annotations (status, method, env, success, error, fault)
# ===========================================================================

class TestHttpAnnotations:
    def _run(self, status_code, extra_env=None):
        import asyncio
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(status_code))

        env = {} if extra_env is None else extra_env
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            asyncio.get_event_loop().run_until_complete(mw.dispatch(req, call_next))
        return mock_segment

    def test_200_success_annotation(self):
        seg = self._run(200)
        seg.put_annotation.assert_any_call("request_success", True)

    def test_200_no_error_annotation(self):
        seg = self._run(200)
        calls = [str(c) for c in seg.put_annotation.call_args_list]
        assert not any("'error', True" in c for c in calls)

    def test_400_error_annotation(self):
        seg = self._run(400)
        seg.put_annotation.assert_any_call("error", True)

    def test_500_fault_annotation(self):
        seg = self._run(500)
        seg.put_annotation.assert_any_call("fault", True)

    def test_500_sets_otel_error_status(self):
        self._run(500)
        mock_otel_span.set_status.assert_called()

    def test_env_annotation(self):
        import middleware.xray_tracing_middleware as mmod
        original_env = mmod.ENV
        try:
            mmod.ENV = "prod"
            seg = self._run(200)
            seg.put_annotation.assert_any_call("env", "prod")
        finally:
            mmod.ENV = original_env

    def test_http_method_annotation(self):
        seg = self._run(200)
        seg.put_annotation.assert_any_call("http_method", "GET")

    def test_http_status_code_annotation(self):
        seg = self._run(200)
        seg.put_annotation.assert_any_call("http_status_code", 200)


# ===========================================================================
# 6. safe_put_annotation – internal edge cases
# ===========================================================================

class TestSafePutAnnotation:
    """
    We exercise safe_put_annotation indirectly by making the mock entity
    raise specific exceptions during put_annotation().
    """

    @pytest.mark.asyncio
    async def test_none_value_skipped(self):
        """None values must not call put_annotation."""
        mw = make_middleware()
        req = make_request(state_attrs={"user": {"sub": None}})
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)

        # user_id=None → safe_put_annotation returns False without calling put_annotation
        annotation_keys = [c.args[0] for c in mock_segment.put_annotation.call_args_list]
        assert "user_id" not in annotation_keys

    @pytest.mark.asyncio
    async def test_already_ended_exception_suppressed(self):
        """AlreadyEndedException must be caught silently."""
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_segment.put_annotation.side_effect = AlreadyEndedException("ended")
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            # Should NOT raise
            await mw.dispatch(req, call_next)

    @pytest.mark.asyncio
    async def test_segment_not_found_exception_suppressed(self):
        """SegmentNotFoundException must be caught silently."""
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_segment.put_annotation.side_effect = SegmentNotFoundException("missing")
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)

    @pytest.mark.asyncio
    async def test_generic_exception_suppressed(self):
        """Any other exception in put_annotation must be caught."""
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_segment.put_annotation.side_effect = RuntimeError("boom")
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)


# ===========================================================================
# 7. safe_span_attr – internal edge cases
# ===========================================================================

class TestSafeSpanAttr:
    @pytest.mark.asyncio
    async def test_none_value_not_set(self):
        mock_otel_span.is_recording.return_value = True
        mw = make_middleware()
        req = make_request(state_attrs={"user": {"email": None}})
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)

        # email None → set_attribute("user.email", ...) must NOT be called with None
        for call in mock_otel_span.set_attribute.call_args_list:
            if call.args[0] == "user.email":
                assert call.args[1] is not None

    @pytest.mark.asyncio
    async def test_span_not_recording_skipped(self):
        mock_otel_span.is_recording.return_value = False
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)

        mock_otel_span.set_attribute.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_span_skipped(self):
        otel_trace.get_current_span.return_value = None
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)
        # No AttributeError should have been raised

    @pytest.mark.asyncio
    async def test_span_attr_exception_suppressed(self):
        mock_otel_span.is_recording.return_value = True
        mock_otel_span.set_attribute.side_effect = Exception("otel boom")
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)
        # Must not raise


# ===========================================================================
# 8. User context capture
# ===========================================================================

class TestUserContext:
    def _dispatch(self, state_attrs):
        import asyncio
        mw = make_middleware()
        req = make_request(state_attrs=state_attrs)
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            asyncio.get_event_loop().run_until_complete(mw.dispatch(req, call_next))
        return mock_segment

    def test_user_id_from_sub(self):
        seg = self._dispatch({"user": {"sub": "u-123"}})
        seg.put_annotation.assert_any_call("user_id", "u-123")

    def test_user_email_from_email_field(self):
        seg = self._dispatch({"user": {"sub": "u-1", "email": "a@b.com"}})
        seg.put_annotation.assert_any_call("user_email", "a@b.com")

    def test_user_email_fallback_preferred_username(self):
        seg = self._dispatch({"user": {"sub": "u-1", "preferred_username": "alice@x.com"}})
        # preferred_username is used as email when email is absent
        seg.put_annotation.assert_any_call("user_email", "alice@x.com")

    def test_username_annotation(self):
        seg = self._dispatch({"user": {"sub": "u-1", "preferred_username": "alice"}})
        seg.put_annotation.assert_any_call("username", "alice")

    def test_realm_extracted_from_issuer(self):
        seg = self._dispatch({"user": {"sub": "u-1", "iss": "https://auth.example.com/realms/myrealm"}})
        seg.put_annotation.assert_any_call("realm", "myrealm")

    def test_issuer_without_slash(self):
        seg = self._dispatch({"user": {"sub": "u-1", "iss": "myrealm"}})
        seg.put_annotation.assert_any_call("realm", "myrealm")

    def test_client_id_annotation(self):
        seg = self._dispatch({"user": {"sub": "u-1", "azp": "my-client"}})
        seg.put_annotation.assert_any_call("client_id", "my-client")

    def test_auth_mode_keycloak(self):
        seg = self._dispatch({"user": {"sub": "u-1"}})
        seg.put_annotation.assert_any_call("auth_mode", "keycloak")

    def test_fallback_user_id_from_state(self):
        seg = self._dispatch({"user_id": "fallback-uid"})
        seg.put_annotation.assert_any_call("user_id", "fallback-uid")

    def test_fallback_user_email_from_state(self):
        seg = self._dispatch({"user_email": "fb@mail.com"})
        seg.put_annotation.assert_any_call("user_email", "fb@mail.com")

    def test_no_user_context_no_error(self):
        """When no user info is present, dispatch should still succeed."""
        seg = self._dispatch({})
        # Basic HTTP annotation still happens
        seg.put_annotation.assert_any_call("http_status_code", 200)


# ===========================================================================
# 9. Error details and agent / message_id capture
# ===========================================================================

class TestExtraAnnotations:
    def _dispatch(self, state_attrs, headers=None):
        import asyncio
        mw = make_middleware()
        req = make_request(state_attrs=state_attrs, headers=headers or {})
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            asyncio.get_event_loop().run_until_complete(mw.dispatch(req, call_next))
        return mock_segment

    def test_error_details_annotated(self):
        seg = self._dispatch({"error_details": {"code": "E001", "error_key": "NOT_FOUND"}})
        seg.put_annotation.assert_any_call("error_code", "E001")
        seg.put_annotation.assert_any_call("error_key", "NOT_FOUND")

    def test_agent_id_annotated(self):
        seg = self._dispatch({"agent_id": "agent-42"})
        seg.put_annotation.assert_any_call("agent_id", "agent-42")

    def test_agent_name_annotated(self):
        seg = self._dispatch({"agent_name": "my-agent"})
        seg.put_annotation.assert_any_call("agent_name", "my-agent")

    def test_message_id_from_state(self):
        seg = self._dispatch({"message_id": "msg-99"})
        seg.put_annotation.assert_any_call("message_id", "msg-99")

    def test_message_id_from_header(self):
        seg = self._dispatch({}, headers={"x-message-id": "hdr-msg-1"})
        seg.put_annotation.assert_any_call("message_id", "hdr-msg-1")

    def test_message_id_state_takes_priority_over_header(self):
        seg = self._dispatch({"message_id": "state-msg"}, headers={"x-message-id": "hdr-msg"})
        keys_vals = [(c.args[0], c.args[1]) for c in seg.put_annotation.call_args_list]
        assert ("message_id", "state-msg") in keys_vals


# ===========================================================================
# 10. put_http_meta exception is handled gracefully
# ===========================================================================

class TestHttpMetaException:
    @pytest.mark.asyncio
    async def test_put_http_meta_exception_does_not_crash(self):
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_segment.put_http_meta.side_effect = Exception("meta fail")
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)
        # Must not raise


# ===========================================================================
# 11. Exception handling block (call_next raises)
# ===========================================================================

class TestExceptionHandling:
    def _setup_exception(self):
        """Return a clean segment mock and reset otel span."""
        mock_otel_span.reset_mock()
        mock_otel_span.is_recording.return_value = True
        mock_otel_span.set_attribute.side_effect = None  # ensure no leftover side_effect
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        return mock_segment

    @pytest.mark.asyncio
    async def test_exception_in_call_next_reraises(self):
        mw = make_middleware()
        req = make_request()
        mock_segment = self._setup_exception()
        call_next = AsyncMock(side_effect=ValueError("downstream error"))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            with pytest.raises(ValueError, match="downstream error"):
                await mw.dispatch(req, call_next)

    @pytest.mark.asyncio
    async def test_exception_annotated_on_segment(self):
        mw = make_middleware()
        req = make_request()
        mock_segment = self._setup_exception()
        call_next = AsyncMock(side_effect=RuntimeError("fail"))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            with pytest.raises(RuntimeError):
                await mw.dispatch(req, call_next)

        mock_segment.put_annotation.assert_any_call("error", True)
        mock_segment.put_annotation.assert_any_call("fault", True)

    @pytest.mark.asyncio
    async def test_exception_type_annotation(self):
        mw = make_middleware()
        req = make_request()
        mock_segment = self._setup_exception()
        call_next = AsyncMock(side_effect=TypeError("type err"))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            with pytest.raises(TypeError):
                await mw.dispatch(req, call_next)

        mock_segment.put_annotation.assert_any_call("error_type", "TypeError")

    @pytest.mark.asyncio
    async def test_add_exception_called_on_segment(self):
        mw = make_middleware()
        req = make_request()
        mock_segment = self._setup_exception()
        call_next = AsyncMock(side_effect=RuntimeError("fail"))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            with pytest.raises(RuntimeError):
                await mw.dispatch(req, call_next)

        mock_segment.add_exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_otel_span_records_exception(self):
        mw = make_middleware()
        req = make_request()
        mock_segment = self._setup_exception()
        call_next = AsyncMock(side_effect=RuntimeError("fail"))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            with pytest.raises(RuntimeError):
                await mw.dispatch(req, call_next)

        mock_otel_span.record_exception.assert_called_once()
        mock_otel_span.set_status.assert_called()

    @pytest.mark.asyncio
    async def test_exception_handling_when_segment_annotation_fails(self):
        """Exception in the exception-handler annotation must not suppress the original error."""
        mw = make_middleware()
        req = make_request()
        mock_otel_span.reset_mock()
        mock_otel_span.is_recording.return_value = True
        mock_otel_span.set_attribute.side_effect = None
        mock_segment = MagicMock()
        mock_segment.put_annotation.side_effect = Exception("annotation fail")
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(side_effect=ValueError("original"))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            with pytest.raises(ValueError, match="original"):
                await mw.dispatch(req, call_next)

    @pytest.mark.asyncio
    async def test_exception_otel_span_not_recording(self):
        mw = make_middleware()
        req = make_request()
        mock_otel_span.reset_mock()
        mock_otel_span.is_recording.return_value = False
        mock_otel_span.set_attribute.side_effect = None
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(side_effect=RuntimeError("fail"))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            with pytest.raises(RuntimeError):
                await mw.dispatch(req, call_next)

        mock_otel_span.record_exception.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_with_no_active_entity(self):
        """When begin_segment returns a mock that lacks .id, exception path should not crash
        because active_entity is falsy (None-like) or because the exception block handles it."""
        mw = make_middleware()
        req = make_request()
        mock_otel_span.reset_mock()
        mock_otel_span.is_recording.return_value = True
        mock_otel_span.set_attribute.side_effect = None
        mock_recorder.begin_segment.side_effect = None
        # Return a segment that has no put_annotation (but won't crash)
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(side_effect=RuntimeError("fail"))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            with pytest.raises(RuntimeError):
                await mw.dispatch(req, call_next)
        # active_entity is the segment, so put_annotation was attempted
        mock_segment.put_annotation.assert_any_call("error", True)


# ===========================================================================
# 12. Finally / cleanup edge cases
# ===========================================================================

class TestFinallyCleanup:
    @pytest.mark.asyncio
    async def test_end_segment_exception_suppressed(self):
        """If end_segment raises, it must be caught and not propagate."""
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        mock_recorder.end_segment.side_effect = Exception("end fail")
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)  # must not raise

    @pytest.mark.asyncio
    async def test_end_subsegment_exception_suppressed(self):
        mw = make_middleware()
        req = make_request()
        mock_subseg = MagicMock()
        mock_recorder.begin_subsegment.return_value = mock_subseg
        mock_recorder.end_subsegment.side_effect = Exception("end sub fail")
        call_next = AsyncMock(return_value=make_response(200))

        with patch.dict(os.environ, {"AWS_LAMBDA_FUNCTION_NAME": "fn"}):
            await mw.dispatch(req, call_next)  # must not raise

    @pytest.mark.asyncio
    async def test_segment_ended_once_even_on_exception(self):
        mock_otel_span.reset_mock()
        mock_otel_span.is_recording.return_value = True
        mock_otel_span.set_attribute.side_effect = None
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        call_next = AsyncMock(side_effect=RuntimeError("fail"))

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            with pytest.raises(RuntimeError):
                await mw.dispatch(req, call_next)

        # end_segment must be called exactly once
        assert mock_recorder.end_segment.call_count == 1


# ===========================================================================
# 13. Response has no status_code attribute
# ===========================================================================

class TestMissingStatusCode:
    @pytest.mark.asyncio
    async def test_defaults_to_500_when_no_status_code(self):
        mock_recorder.begin_segment.side_effect = None
        mock_otel_span.set_attribute.side_effect = None
        mw = make_middleware()
        req = make_request()
        mock_segment = MagicMock()
        mock_recorder.begin_segment.return_value = mock_segment
        resp = MagicMock(spec=[])  # no status_code attribute
        call_next = AsyncMock(return_value=resp)

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_LAMBDA_FUNCTION_NAME", None)
            await mw.dispatch(req, call_next)

        # status_code defaults to 500 → error + fault annotations
        mock_segment.put_annotation.assert_any_call("error", True)
        mock_segment.put_annotation.assert_any_call("fault", True)