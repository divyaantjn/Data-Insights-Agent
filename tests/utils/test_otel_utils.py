"""test_otel_utils.py — 95%+ coverage for utils/otel_utils.py

Run from project root:
    pytest tests/test_otel_utils.py -v

Works on Windows, macOS, and Linux — uses importlib for OS-agnostic loading.
"""

import sys
import os
import types
import importlib
import importlib.util
import unittest.mock as _um
from unittest.mock import MagicMock, patch, call
import pytest

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# OS-agnostic path resolution
_SPEC_FINDER = importlib.util.find_spec("src.utils.otel_utils")
if _SPEC_FINDER is not None:
    _SRC_PATH = _SPEC_FINDER.origin
else:
    _SRC_PATH = os.path.join(_ROOT, "src.utils", "otel_utils.py")


# =============================================================================
# Stub builder
# =============================================================================

def _build_stubs():
    """
    Build and register stubs for aws_xray_sdk and opentelemetry so the module
    can be imported without those packages installed.
    Returns a dict of key stub objects for test use.
    """

    # ── aws_xray_sdk exceptions ───────────────────────────────────────────────
    class _AlreadyEndedException(Exception):
        pass

    class _SegmentNotFoundException(Exception):
        pass

    # ── xray_recorder stub ────────────────────────────────────────────────────
    _segment  = MagicMock(name="segment")
    _recorder = MagicMock(name="xray_recorder")
    _recorder.current_segment.return_value = _segment

    # ── aws_xray_sdk package hierarchy ────────────────────────────────────────
    _xray_pkg              = types.ModuleType("aws_xray_sdk")
    _xray_pkg.__path__     = []
    _xray_pkg.__package__  = "aws_xray_sdk"

    _xray_core             = types.ModuleType("aws_xray_sdk.core")
    _xray_core.__path__    = []
    _xray_core.__package__ = "aws_xray_sdk"
    _xray_core.xray_recorder = _recorder

    _xray_exc_pkg              = types.ModuleType("aws_xray_sdk.core.exceptions")
    _xray_exc_pkg.__path__     = []
    _xray_exc_pkg.__package__  = "aws_xray_sdk.core"

    _xray_exc_mod = types.ModuleType("aws_xray_sdk.core.exceptions.exceptions")
    _xray_exc_mod.__package__        = "aws_xray_sdk.core.exceptions"
    _xray_exc_mod.AlreadyEndedException    = _AlreadyEndedException
    _xray_exc_mod.SegmentNotFoundException = _SegmentNotFoundException

    # ── opentelemetry stubs ───────────────────────────────────────────────────
    _otel_pkg             = types.ModuleType("opentelemetry")
    _otel_pkg.__path__    = []
    _otel_pkg.__package__ = "opentelemetry"

    _otel_span = MagicMock(name="span")
    _otel_span.is_recording.return_value = True

    _otel_trace             = types.ModuleType("opentelemetry.trace")
    _otel_trace.__package__ = "opentelemetry"
    _otel_trace.get_current_span = MagicMock(return_value=_otel_span)

    # ── register everything ───────────────────────────────────────────────────
    _mods = {
        "aws_xray_sdk":                             _xray_pkg,
        "aws_xray_sdk.core":                        _xray_core,
        "aws_xray_sdk.core.exceptions":             _xray_exc_pkg,
        "aws_xray_sdk.core.exceptions.exceptions":  _xray_exc_mod,
        "opentelemetry":                            _otel_pkg,
        "opentelemetry.trace":                      _otel_trace,
    }
    for k, v in _mods.items():
        sys.modules.setdefault(k, v)

    return {
        "recorder":                   _recorder,
        "segment":                    _segment,
        "span":                       _otel_span,
        "otel_trace":                 _otel_trace,
        "AlreadyEndedException":      _AlreadyEndedException,
        "SegmentNotFoundException":   _SegmentNotFoundException,
    }


_STUBS = _build_stubs()


# =============================================================================
# Fresh module loader
# =============================================================================

def _load_fresh() -> types.ModuleType:
    """
    Load a completely fresh copy of otel_utils, resetting all mocks first so
    test mutations in one test cannot bleed into another.

    Key fix: after reset_mock(), explicitly restore all return_values and
    side_effects because reset_mock() wipes configured return values.
    """
    # ── Reset shared mocks ────────────────────────────────────────────────────
    # Use reset_mock(return_value=False) to preserve return_value config,
    # then clear side_effects explicitly.  This avoids the bleed where a test
    # sets side_effect but forgets to clear it, and also avoids reset_mock()
    # wiping return_value and leaving current_segment() returning a bare Mock
    # instead of _STUBS["segment"].

    _STUBS["recorder"].reset_mock()
    _STUBS["segment"].reset_mock()
    _STUBS["span"].reset_mock()

    # Restore return values that reset_mock() may have cleared
    _STUBS["recorder"].current_segment.return_value = _STUBS["segment"]
    _STUBS["recorder"].current_segment.side_effect  = None

    _STUBS["segment"].put_annotation.side_effect  = None
    _STUBS["segment"].put_metadata.side_effect    = None

    _STUBS["span"].is_recording.return_value = True
    _STUBS["span"].is_recording.side_effect  = None
    _STUBS["span"].set_attribute.side_effect = None

    # Restore get_current_span on the shared trace module object so that the
    # freshly-loaded module (which holds a reference to that same module object
    # and calls trace.get_current_span() at runtime) always gets _STUBS["span"].
    _otel_trace = sys.modules["opentelemetry.trace"]
    _otel_trace.get_current_span = MagicMock(return_value=_STUBS["span"])

    # Restore xray_recorder on the shared core module
    _xray_core = sys.modules["aws_xray_sdk.core"]
    _xray_core.xray_recorder = _STUBS["recorder"]

    # ── Load fresh module via importlib ───────────────────────────────────────
    spec = importlib.util.spec_from_file_location(
        "_otel_utils_fresh", _SRC_PATH, submodule_search_locations=[]
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "utils"
    spec.loader.exec_module(mod)

    # The freshly loaded module now holds its OWN reference to the
    # opentelemetry.trace module object (via `from opentelemetry import trace`).
    # That object IS sys.modules["opentelemetry.trace"], so patching
    # get_current_span on it (above) is visible to the module at call time.
    # No further action needed here.

    return mod


def _fresh():
    return _load_fresh()


# =============================================================================
# set_xray_annotation
# =============================================================================

class TestSetXrayAnnotation:

    # ── None value ────────────────────────────────────────────────────────────

    def test_none_value_returns_immediately(self):
        m = _fresh()
        m.set_xray_annotation("key", None)
        _STUBS["segment"].put_annotation.assert_not_called()

    # ── Type coercions ────────────────────────────────────────────────────────

    def test_bool_true_passed_as_bool(self):
        m = _fresh()
        m.set_xray_annotation("flag", True)
        _STUBS["segment"].put_annotation.assert_called_once_with("flag", True)

    def test_bool_false_passed_as_bool(self):
        m = _fresh()
        m.set_xray_annotation("flag", False)
        _STUBS["segment"].put_annotation.assert_called_once_with("flag", False)

    def test_int_value_passed_as_int(self):
        m = _fresh()
        m.set_xray_annotation("count", 42)
        _STUBS["segment"].put_annotation.assert_called_once_with("count", 42)

    def test_float_value_passed_as_float(self):
        m = _fresh()
        m.set_xray_annotation("score", 3.14)
        _STUBS["segment"].put_annotation.assert_called_once_with("score", 3.14)

    def test_string_value_passed_as_string(self):
        m = _fresh()
        m.set_xray_annotation("name", "alice")
        _STUBS["segment"].put_annotation.assert_called_once_with("name", "alice")

    def test_string_truncated_to_500_chars(self):
        m   = _fresh()
        long_val = "x" * 600
        m.set_xray_annotation("k", long_val)
        args = _STUBS["segment"].put_annotation.call_args[0]
        assert len(args[1]) == 500

    def test_unknown_type_converted_to_str(self):
        m = _fresh()

        class _Custom:
            def __str__(self):
                return "custom_repr"

        m.set_xray_annotation("obj", _Custom())
        args = _STUBS["segment"].put_annotation.call_args[0]
        assert args[1] == "custom_repr"

    def test_unknown_type_truncated_to_500(self):
        m = _fresh()

        class _Big:
            def __str__(self):
                return "z" * 600

        m.set_xray_annotation("big", _Big())
        args = _STUBS["segment"].put_annotation.call_args[0]
        assert len(args[1]) == 500

    # ── Key sanitisation ──────────────────────────────────────────────────────

    def test_dot_in_key_replaced_with_underscore(self):
        m = _fresh()
        m.set_xray_annotation("user.id", "u1")
        args = _STUBS["segment"].put_annotation.call_args[0]
        assert args[0] == "user_id"

    def test_multiple_dots_in_key_all_replaced(self):
        m = _fresh()
        m.set_xray_annotation("a.b.c", "v")
        args = _STUBS["segment"].put_annotation.call_args[0]
        assert args[0] == "a_b_c"

    def test_key_truncated_to_500_chars(self):
        m   = _fresh()
        long_key = "k" * 600
        m.set_xray_annotation(long_key, "v")
        args = _STUBS["segment"].put_annotation.call_args[0]
        assert len(args[0]) == 500

    def test_key_without_dot_unchanged(self):
        m = _fresh()
        m.set_xray_annotation("plain_key", "v")
        args = _STUBS["segment"].put_annotation.call_args[0]
        assert args[0] == "plain_key"

    # ── No segment ────────────────────────────────────────────────────────────

    def test_no_segment_does_not_call_put_annotation(self):
        m = _fresh()
        _STUBS["recorder"].current_segment.return_value = None
        m.set_xray_annotation("k", "v")
        _STUBS["segment"].put_annotation.assert_not_called()

    def test_no_segment_logs_warning(self, caplog):
        import logging
        m = _fresh()
        _STUBS["recorder"].current_segment.return_value = None
        with caplog.at_level(logging.WARNING):
            m.set_xray_annotation("mykey", "v")
        assert "mykey" in caplog.text

    # ── Exception handling ────────────────────────────────────────────────────

    def test_already_ended_exception_does_not_raise(self):
        m = _fresh()
        _STUBS["recorder"].current_segment.side_effect = (
            _STUBS["AlreadyEndedException"]("ended"))
        m.set_xray_annotation("k", "v")   # must not raise
        _STUBS["recorder"].current_segment.side_effect = None

    def test_segment_not_found_exception_does_not_raise(self):
        m = _fresh()
        _STUBS["recorder"].current_segment.side_effect = (
            _STUBS["SegmentNotFoundException"]("not found"))
        m.set_xray_annotation("k", "v")   # must not raise
        _STUBS["recorder"].current_segment.side_effect = None

    def test_already_ended_on_put_annotation_does_not_raise(self):
        m = _fresh()
        _STUBS["segment"].put_annotation.side_effect = (
            _STUBS["AlreadyEndedException"]("ended"))
        m.set_xray_annotation("k", "v")   # must not raise
        _STUBS["segment"].put_annotation.side_effect = None

    def test_generic_exception_does_not_raise(self):
        m = _fresh()
        _STUBS["recorder"].current_segment.side_effect = RuntimeError("boom")
        m.set_xray_annotation("k", "v")   # must not raise
        _STUBS["recorder"].current_segment.side_effect = None

    def test_generic_exception_on_put_annotation_does_not_raise(self):
        m = _fresh()
        _STUBS["segment"].put_annotation.side_effect = RuntimeError("boom")
        m.set_xray_annotation("k", "v")   # must not raise
        _STUBS["segment"].put_annotation.side_effect = None

    def test_generic_exception_logs_warning(self, caplog):
        import logging
        m = _fresh()
        _STUBS["segment"].put_annotation.side_effect = RuntimeError("oops")
        with caplog.at_level(logging.WARNING):
            m.set_xray_annotation("k", "v")
        assert "k" in caplog.text
        _STUBS["segment"].put_annotation.side_effect = None

    # ── Happy path logging ────────────────────────────────────────────────────

    def test_successful_annotation_logs_debug(self, caplog):
        import logging
        m = _fresh()
        with caplog.at_level(logging.DEBUG):
            m.set_xray_annotation("env", "prod")
        assert "env" in caplog.text or "ANNOTATION" in caplog.text


# =============================================================================
# set_xray_metadata
# =============================================================================

class TestSetXrayMetadata:

    # ── None value ────────────────────────────────────────────────────────────

    def test_none_value_returns_immediately(self):
        m = _fresh()
        m.set_xray_metadata("ns", "key", None)
        _STUBS["segment"].put_metadata.assert_not_called()

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_string_value_sent_to_put_metadata(self):
        m = _fresh()
        m.set_xray_metadata("app", "version", "1.0")
        _STUBS["segment"].put_metadata.assert_called_once_with(
            "version", "1.0", "app")

    def test_dict_value_sent_to_put_metadata(self):
        m = _fresh()
        payload = {"a": 1, "b": [2, 3]}
        m.set_xray_metadata("ns", "data", payload)
        _STUBS["segment"].put_metadata.assert_called_once_with(
            "data", payload, "ns")

    def test_int_value_sent_to_put_metadata(self):
        m = _fresh()
        m.set_xray_metadata("ns", "count", 99)
        _STUBS["segment"].put_metadata.assert_called_once_with("count", 99, "ns")

    def test_list_value_sent_to_put_metadata(self):
        m = _fresh()
        m.set_xray_metadata("ns", "items", [1, 2, 3])
        _STUBS["segment"].put_metadata.assert_called_once_with(
            "items", [1, 2, 3], "ns")

    def test_namespace_and_key_passed_correctly(self):
        m = _fresh()
        m.set_xray_metadata("my_namespace", "my_key", "val")
        args = _STUBS["segment"].put_metadata.call_args[0]
        assert args[0] == "my_key"
        assert args[2] == "my_namespace"

    # ── No segment ────────────────────────────────────────────────────────────

    def test_no_segment_does_not_call_put_metadata(self):
        m = _fresh()
        _STUBS["recorder"].current_segment.return_value = None
        m.set_xray_metadata("ns", "k", "v")
        _STUBS["segment"].put_metadata.assert_not_called()

    def test_no_segment_logs_warning(self, caplog):
        import logging
        m = _fresh()
        _STUBS["recorder"].current_segment.return_value = None
        with caplog.at_level(logging.WARNING):
            m.set_xray_metadata("my_ns", "my_key", "v")
        assert "my_ns" in caplog.text or "my_key" in caplog.text

    # ── Exception handling ────────────────────────────────────────────────────

    def test_already_ended_exception_does_not_raise(self):
        m = _fresh()
        _STUBS["recorder"].current_segment.side_effect = (
            _STUBS["AlreadyEndedException"]("ended"))
        m.set_xray_metadata("ns", "k", "v")
        _STUBS["recorder"].current_segment.side_effect = None

    def test_segment_not_found_exception_does_not_raise(self):
        m = _fresh()
        _STUBS["recorder"].current_segment.side_effect = (
            _STUBS["SegmentNotFoundException"]("nf"))
        m.set_xray_metadata("ns", "k", "v")
        _STUBS["recorder"].current_segment.side_effect = None

    def test_already_ended_on_put_metadata_does_not_raise(self):
        m = _fresh()
        _STUBS["segment"].put_metadata.side_effect = (
            _STUBS["AlreadyEndedException"]("ended"))
        m.set_xray_metadata("ns", "k", "v")
        _STUBS["segment"].put_metadata.side_effect = None

    def test_segment_not_found_on_put_metadata_does_not_raise(self):
        m = _fresh()
        _STUBS["segment"].put_metadata.side_effect = (
            _STUBS["SegmentNotFoundException"]("nf"))
        m.set_xray_metadata("ns", "k", "v")
        _STUBS["segment"].put_metadata.side_effect = None

    def test_generic_exception_does_not_raise(self):
        m = _fresh()
        _STUBS["recorder"].current_segment.side_effect = RuntimeError("boom")
        m.set_xray_metadata("ns", "k", "v")
        _STUBS["recorder"].current_segment.side_effect = None

    def test_generic_exception_on_put_metadata_does_not_raise(self):
        m = _fresh()
        _STUBS["segment"].put_metadata.side_effect = RuntimeError("boom")
        m.set_xray_metadata("ns", "k", "v")
        _STUBS["segment"].put_metadata.side_effect = None

    def test_generic_exception_logs_warning(self, caplog):
        import logging
        m = _fresh()
        _STUBS["segment"].put_metadata.side_effect = ValueError("bad")
        with caplog.at_level(logging.WARNING):
            m.set_xray_metadata("ns", "k", "v")
        assert "ns" in caplog.text or "k" in caplog.text
        _STUBS["segment"].put_metadata.side_effect = None

    # ── Happy path logging ────────────────────────────────────────────────────

    def test_successful_metadata_logs_debug(self, caplog):
        import logging
        m = _fresh()
        with caplog.at_level(logging.DEBUG):
            m.set_xray_metadata("ns", "key", "val")
        assert "ns" in caplog.text or "METADATA" in caplog.text


# =============================================================================
# force_user_context_to_xray
# =============================================================================

class TestForceUserContextToXray:

    # ── X-Ray annotation calls ────────────────────────────────────────────────

    def test_user_id_sets_xray_annotation(self):
        m = _fresh()
        m.force_user_context_to_xray(user_id="u123")
        calls = [c[0][0] for c in _STUBS["segment"].put_annotation.call_args_list]
        assert "user_id" in calls

    def test_user_id_value_is_str(self):
        m = _fresh()
        m.force_user_context_to_xray(user_id=999)   # integer user_id → str
        calls = {c[0][0]: c[0][1]
                 for c in _STUBS["segment"].put_annotation.call_args_list}
        assert calls["user_id"] == "999"

    def test_user_email_sets_xray_annotation(self):
        m = _fresh()
        m.force_user_context_to_xray(user_email="a@b.com")
        calls = [c[0][0] for c in _STUBS["segment"].put_annotation.call_args_list]
        assert "user_email" in calls

    def test_auth_mode_sets_xray_annotation(self):
        m = _fresh()
        m.force_user_context_to_xray(auth_mode="keycloak")
        calls = [c[0][0] for c in _STUBS["segment"].put_annotation.call_args_list]
        assert "auth_mode" in calls

    def test_all_three_fields_set_three_annotations(self):
        m = _fresh()
        m.force_user_context_to_xray(
            user_id="u1", user_email="u@e.com", auth_mode="api_key")
        assert _STUBS["segment"].put_annotation.call_count == 3

    def test_no_args_sets_no_annotations(self):
        m = _fresh()
        m.force_user_context_to_xray()
        _STUBS["segment"].put_annotation.assert_not_called()

    def test_only_user_id_sets_one_annotation(self):
        m = _fresh()
        m.force_user_context_to_xray(user_id="u1")
        assert _STUBS["segment"].put_annotation.call_count == 1

    def test_only_user_email_sets_one_annotation(self):
        m = _fresh()
        m.force_user_context_to_xray(user_email="x@y.com")
        assert _STUBS["segment"].put_annotation.call_count == 1

    def test_only_auth_mode_sets_one_annotation(self):
        m = _fresh()
        m.force_user_context_to_xray(auth_mode="api_key")
        assert _STUBS["segment"].put_annotation.call_count == 1

    # ── OTEL span attribute calls ─────────────────────────────────────────────

    def test_user_id_sets_otel_span_attribute(self):
        m = _fresh()
        # Patch get_current_span on the module's own trace reference
        m.trace.get_current_span = MagicMock(return_value=_STUBS["span"])
        m.force_user_context_to_xray(user_id="u1")
        _STUBS["span"].set_attribute.assert_any_call("user.id", "u1")

    def test_user_email_sets_otel_span_attribute(self):
        m = _fresh()
        m.trace.get_current_span = MagicMock(return_value=_STUBS["span"])
        m.force_user_context_to_xray(user_email="a@b.com")
        _STUBS["span"].set_attribute.assert_any_call("user.email", "a@b.com")

    def test_auth_mode_sets_otel_span_attribute(self):
        m = _fresh()
        m.trace.get_current_span = MagicMock(return_value=_STUBS["span"])
        m.force_user_context_to_xray(auth_mode="keycloak")
        _STUBS["span"].set_attribute.assert_any_call("auth.mode", "keycloak")

    def test_all_three_fields_set_three_otel_attributes(self):
        m = _fresh()
        m.trace.get_current_span = MagicMock(return_value=_STUBS["span"])
        m.force_user_context_to_xray(
            user_id="u1", user_email="u@e.com", auth_mode="api_key")
        assert _STUBS["span"].set_attribute.call_count == 3

    def test_no_args_sets_no_otel_attributes(self):
        m = _fresh()
        m.trace.get_current_span = MagicMock(return_value=_STUBS["span"])
        m.force_user_context_to_xray()
        _STUBS["span"].set_attribute.assert_not_called()

    def test_user_id_integer_cast_to_str_for_otel(self):
        m = _fresh()
        m.trace.get_current_span = MagicMock(return_value=_STUBS["span"])
        m.force_user_context_to_xray(user_id=42)
        _STUBS["span"].set_attribute.assert_any_call("user.id", "42")

    # ── Span not recording ────────────────────────────────────────────────────

    def test_span_not_recording_skips_otel_attributes(self):
        m = _fresh()
        _STUBS["span"].is_recording.return_value = False
        m.trace.get_current_span = MagicMock(return_value=_STUBS["span"])
        m.force_user_context_to_xray(
            user_id="u1", user_email="u@e.com", auth_mode="kc")
        _STUBS["span"].set_attribute.assert_not_called()

    def test_span_not_recording_still_sets_xray_annotations(self):
        m = _fresh()
        _STUBS["span"].is_recording.return_value = False
        m.trace.get_current_span = MagicMock(return_value=_STUBS["span"])
        m.force_user_context_to_xray(user_id="u1")
        assert _STUBS["segment"].put_annotation.call_count == 1

    # ── No span (get_current_span returns None) ───────────────────────────────

    def test_no_otel_span_does_not_raise(self):
        m = _fresh()
        m.trace.get_current_span = MagicMock(return_value=None)
        m.force_user_context_to_xray(user_id="u1")   # must not raise

    def test_no_otel_span_still_sets_xray_annotations(self):
        m = _fresh()
        m.trace.get_current_span = MagicMock(return_value=None)
        m.force_user_context_to_xray(user_id="u1")
        assert _STUBS["segment"].put_annotation.call_count == 1

    # ── X-Ray failures don't block OTEL ──────────────────────────────────────

    def test_xray_annotation_failure_still_sets_otel_attributes(self):
        m = _fresh()
        m.trace.get_current_span = MagicMock(return_value=_STUBS["span"])
        _STUBS["recorder"].current_segment.side_effect = RuntimeError("xray down")
        m.force_user_context_to_xray(user_id="u1", user_email="u@e.com",
                                      auth_mode="kc")
        assert _STUBS["span"].set_attribute.call_count == 3
        _STUBS["recorder"].current_segment.side_effect = None

    # ── Empty-string fields treated as falsy ──────────────────────────────────

    def test_empty_user_id_skips_annotation(self):
        m = _fresh()
        m.force_user_context_to_xray(user_id="")
        calls = [c[0][0] for c in _STUBS["segment"].put_annotation.call_args_list]
        assert "user_id" not in calls

    def test_empty_user_email_skips_annotation(self):
        m = _fresh()
        m.force_user_context_to_xray(user_email="")
        calls = [c[0][0] for c in _STUBS["segment"].put_annotation.call_args_list]
        assert "user_email" not in calls

    def test_empty_auth_mode_skips_annotation(self):
        m = _fresh()
        m.force_user_context_to_xray(auth_mode="")
        calls = [c[0][0] for c in _STUBS["segment"].put_annotation.call_args_list]
        assert "auth_mode" not in calls

    # ── Correct annotation values ─────────────────────────────────────────────

    def test_user_email_value_correct(self):
        m = _fresh()
        m.force_user_context_to_xray(user_email="test@example.com")
        calls = {c[0][0]: c[0][1]
                 for c in _STUBS["segment"].put_annotation.call_args_list}
        assert calls["user_email"] == "test@example.com"

    def test_auth_mode_value_correct(self):
        m = _fresh()
        m.force_user_context_to_xray(auth_mode="api_key")
        calls = {c[0][0]: c[0][1]
                 for c in _STUBS["segment"].put_annotation.call_args_list}
        assert calls["auth_mode"] == "api_key"