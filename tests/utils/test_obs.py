import pytest
from unittest.mock import patch, MagicMock
import types

from src.utils.obs import LLMUsageTracker, observe_token_usage


# -----------------------------
# Helpers
# -----------------------------

class DummyResponse:
    def __init__(self, usage):
        self.usage = usage


class DummyUsageObject:
    def __init__(self, prompt_tokens=1, completion_tokens=2, total_tokens=3):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


# -----------------------------
# Tests for LLMUsageTracker
# -----------------------------

@patch("src.utils.obs.kafka_logger")
@patch("src.utils.obs.extract_user_context")
def test_track_response_success_dict_usage(mock_extract, mock_kafka):
    mock_extract.return_value = {"encrypted_payload": "abc123"}

    tracker = LLMUsageTracker(auth_token="token")

    response = DummyResponse({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15
    })

    result = tracker.track_response(response, model_name="openai/gpt-4")

    assert result["status"] == "success"
    assert result["total_tokens"] == 15

    mock_kafka.log.assert_called_once()
    payload = mock_kafka.log.call_args[0][0]

    assert payload["prompt_tokens"] == 10
    assert payload["completion_tokens"] == 5
    assert payload["total_tokens"] == 15
    assert payload["model_name"] == "gpt-4"


@patch("src.utils.obs.kafka_logger")
@patch("src.utils.obs.extract_user_context")
def test_track_response_usage_object(mock_extract, mock_kafka):
    mock_extract.return_value = {"encrypted_payload": "abc123"}

    tracker = LLMUsageTracker(auth_token="token")

    usage_obj = DummyUsageObject(3, 4, 7)
    response = DummyResponse(usage_obj)

    result = tracker.track_response(response, model_name="model-x")

    assert result["status"] == "success"
    assert result["total_tokens"] == 7

    payload = mock_kafka.log.call_args[0][0]
    assert payload["model_name"] == "model-x"


@patch("src.utils.obs.kafka_logger")
@patch("src.utils.obs.extract_user_context")
def test_track_response_no_usage(mock_extract, mock_kafka):
    tracker = LLMUsageTracker(auth_token="token")

    response = DummyResponse(None)

    result = tracker.track_response(response)

    assert result["status"] == "error"
    assert "No usage info" in result["message"]
    mock_kafka.log.assert_not_called()


@patch("src.utils.obs.kafka_logger")
@patch("src.utils.obs.extract_user_context")
def test_track_response_invalid_usage_type(mock_extract, mock_kafka):
    tracker = LLMUsageTracker(auth_token="token")

    response = DummyResponse("invalid_usage")

    result = tracker.track_response(response)

    assert result["status"] == "error"
    assert "Cannot parse usage" in result["message"]


@patch("src.utils.obs.kafka_logger")
@patch("src.utils.obs.extract_user_context")
def test_track_response_zero_tokens_no_kafka(mock_extract, mock_kafka):
    mock_extract.return_value = {"encrypted_payload": "abc123"}

    tracker = LLMUsageTracker(auth_token="token")

    response = DummyResponse({
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    })

    result = tracker.track_response(response)

    assert result["status"] == "success"
    assert result["total_tokens"] == 0
    mock_kafka.log.assert_not_called()


@patch("src.utils.obs.kafka_logger")
@patch("src.utils.obs.extract_user_context")
def test_track_response_exception(mock_extract, mock_kafka):
    mock_extract.side_effect = Exception("boom")

    tracker = LLMUsageTracker(auth_token="token")

    response = DummyResponse({
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2
    })

    result = tracker.track_response(response)

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_model_name_default_when_none():
    tracker = LLMUsageTracker(auth_token="token")

    with patch("src.utils.obs.extract_user_context") as mock_extract, \
         patch("src.utils.obs.kafka_logger") as mock_kafka:

        mock_extract.return_value = {"encrypted_payload": "abc"}

        response = DummyResponse({
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2
        })

        tracker.track_response(response, model_name=None)

        payload = mock_kafka.log.call_args[0][0]
        assert payload["model_name"] == "UNKNOWN_MODEL"


# -----------------------------
# Tests for observe_token_usage
# -----------------------------

@patch("src.utils.obs.LLMUsageTracker")
def test_observe_token_usage_calls_tracker(mock_tracker_cls):
    mock_tracker = MagicMock()
    mock_tracker_cls.return_value = mock_tracker

    result = DummyResponse({
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2
    })

    observe_token_usage(result, auth_token="token", model_name="abc")

    mock_tracker_cls.assert_called_once_with(auth_token="token")
    mock_tracker.track_response.assert_called_once_with(result, model_name="abc")


@patch("src.utils.obs.LLMUsageTracker")
def test_observe_token_usage_no_auth_token(mock_tracker_cls):
    result = DummyResponse({
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2
    })

    observe_token_usage(result, auth_token=None)

    mock_tracker_cls.assert_not_called()