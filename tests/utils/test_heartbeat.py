import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.utils.heartbeat import (
    HeartbeatClient,
    send_heartbeat_sync,
    send_periodic_heartbeat_sync,
)


# -----------------------------------------------------------
# Fixtures
# -----------------------------------------------------------

@pytest.fixture
def client():
    with patch.dict("os.environ", {
        "LICENSE_BACKEND_URL": "http://test.com",
        "AGENT_UNIQUE_ID": "agent-123",
        "SERVER_NAME": "test-agent",
        "ENV": "test",
        "APP_VERSION": "1.0"
    }):
        yield HeartbeatClient()


# -----------------------------------------------------------
# Test: _build_metadata
# -----------------------------------------------------------

def test_build_metadata_basic(client):
    metadata = client._build_metadata()

    assert metadata["heartbeat_type"] == "periodic"
    assert metadata["agent_unique_id"] == "agent-123"
    assert metadata["agent_name"] == "test-agent"
    assert metadata["environment"] == "test"
    assert metadata["execution_context"] == "background_task"
    assert metadata["version"] == "1.0"


def test_build_metadata_with_api_path(client):
    metadata = client._build_metadata(api_path="/test")

    assert metadata["api_path"] == "/test"
    assert metadata["execution_context"] == "api_request"


def test_build_metadata_override(client):
    metadata = client._build_metadata(metadata={"extra": "value"})

    assert metadata["extra"] == "value"


# -----------------------------------------------------------
# Test: send_heartbeat (async)
# -----------------------------------------------------------

@pytest.mark.asyncio
async def test_send_heartbeat_success(client):
    mock_response = MagicMock(status_code=200)

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        result = await client.send_heartbeat()

    assert result is True


@pytest.mark.asyncio
async def test_send_heartbeat_failure_status(client):
    mock_response = MagicMock(status_code=500)

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        result = await client.send_heartbeat()

    assert result is False


@pytest.mark.asyncio
async def test_send_heartbeat_exception(client):
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=Exception("boom"))):
        result = await client.send_heartbeat()

    assert result is False


# -----------------------------------------------------------
# Test: execution heartbeat
# -----------------------------------------------------------

@pytest.mark.asyncio
async def test_send_execution_heartbeat(client):
    with patch.object(client, "send_heartbeat", new=AsyncMock(return_value=True)) as mock_send:
        result = await client.send_execution_heartbeat("/api/test", {"request_id": "123"})

    assert result is True
    mock_send.assert_called_once()


# -----------------------------------------------------------
# Test: completion heartbeat
# -----------------------------------------------------------

@pytest.mark.asyncio
async def test_send_completion_heartbeat(client):
    with patch.object(client, "send_heartbeat", new=AsyncMock(return_value=True)) as mock_send:
        result = await client.send_completion_heartbeat("/api/test")

    assert result is True
    mock_send.assert_called_once()


# -----------------------------------------------------------
# Test: periodic heartbeat
# -----------------------------------------------------------

@pytest.mark.asyncio
async def test_send_periodic_heartbeat(client):
    with patch.object(client, "send_heartbeat", new=AsyncMock(return_value=True)) as mock_send:
        result = await client.send_periodic_heartbeat()

    assert result is True
    mock_send.assert_called_once()


# -----------------------------------------------------------
# Test: start_periodic_heartbeat
# -----------------------------------------------------------

@pytest.mark.asyncio
async def test_start_periodic_heartbeat():
    from src.utils.heartbeat import start_periodic_heartbeat, heartbeat_client

    with patch.object(
        heartbeat_client, 
        "send_periodic_heartbeat", 
        new=AsyncMock(return_value=True)
    ):
        task = await start_periodic_heartbeat(interval=1)

        await asyncio.sleep(1.5)  # let it run once
        task.cancel()

        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        assert task.cancelled() or task.done()


# -----------------------------------------------------------
# Test: sync heartbeat (requests)
# -----------------------------------------------------------

def test_send_heartbeat_sync_success():
    mock_response = MagicMock(status_code=200)

    with patch.dict("os.environ", {
        "LICENSE_BACKEND_URL": "http://test.com",
        "AGENT_UNIQUE_ID": "agent-123",
        "SERVER_NAME": "test-agent",
        "ENV": "test"
    }):
        with patch("requests.post", return_value=mock_response):
            result = send_heartbeat_sync()

    assert result is True


def test_send_heartbeat_sync_failure():
    mock_response = MagicMock(status_code=500)

    with patch("requests.post", return_value=mock_response):
        result = send_heartbeat_sync()

    assert result is False


def test_send_heartbeat_sync_exception():
    with patch("requests.post", side_effect=Exception("boom")):
        result = send_heartbeat_sync()

    assert result is False


# -----------------------------------------------------------
# Test: periodic sync wrapper
# -----------------------------------------------------------

def test_send_periodic_heartbeat_sync():
    with patch("src.utils.heartbeat.send_heartbeat_sync", return_value=True) as mock_fn:
        result = send_periodic_heartbeat_sync()

    assert result is True
    mock_fn.assert_called_once()