"""
Tests for app.py module.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import FastAPI
from src.api.app import create_app


class TestCreateApp:
    """Tests for create_app function."""

    @patch('src.api.app.setup_opik_tracing', create=True)
    @patch('src.api.app.ConfigLoader')
    def test_create_app_returns_fastapi_instance(self, mock_config, mock_opik):
        """Test that create_app returns FastAPI instance."""
        mock_config.return_value.get_app_config.return_value = {
            'name': 'Test App',
            'version': '1.0.0',
            'debug': False,
            'enable_tracing': True
        }

        app = create_app()

        assert isinstance(app, FastAPI)
        assert app.title == 'Test App'
        assert app.version == '1.0.0'

    @patch('src.api.app.setup_opik_tracing', create=True)
    @patch('src.api.app.ConfigLoader')
    def test_create_app_with_debug_mode(self, mock_config, mock_opik):
        """Test create_app with debug mode enabled."""
        mock_config.return_value.get_app_config.return_value = {
            'name': 'Debug App',
            'version': '2.0.0',
            'debug': True
        }

        app = create_app()

        assert app.debug is True

    @patch('src.api.app.setup_opik_tracing', create=True)
    @patch('src.api.app.ConfigLoader')
    def test_create_app_has_lifespan(self, mock_config, mock_opik):
        """Test that create_app configures lifespan."""
        mock_config.return_value.get_app_config.return_value = {
            'name': 'Test App',
            'version': '1.0.0',
            'debug': False
        }

        app = create_app()

        assert app.router.lifespan_context is not None


class DummyHeartbeatTask:
    def __init__(self):
        self._done = False
        self.cancel_called = False

    def done(self):
        return self._done

    def cancel(self):
        self.cancel_called = True
        self._done = True

    def __await__(self):
        async def _dummy():
            return None
        return _dummy().__await__()


def _make_mock_heartbeat_task():
    return DummyHeartbeatTask()


class TestLifespan:
    """Tests for lifespan context manager."""

    @pytest.mark.asyncio
    @patch('src.api.app.start_periodic_heartbeat')
    @patch('src.api.app.initialize_connection_pool')
    @patch('src.api.app.initialize_database')
    @patch('src.api.app.close_connection_pool')
    @patch('src.api.app.flush_traces')
    @patch.dict('os.environ', {}, clear=False)
    async def test_lifespan_startup_success(
        self, mock_flush, mock_close, mock_init_db, mock_init_pool, mock_heartbeat
    ):
        """Test successful lifespan startup."""
        import os
        import src.api.app as app_module

        os.environ.pop('AWS_LAMBDA_FUNCTION_NAME', None)
        app_module._heartbeat_task = None

        mock_heartbeat.return_value = _make_mock_heartbeat_task()

        from src.api.app import lifespan

        app = Mock()

        async with lifespan(app):
            mock_init_pool.assert_called_once()
            mock_init_db.assert_called_once()
            mock_heartbeat.assert_called_once_with(interval=30)

        mock_flush.assert_called_once()
        mock_close.assert_called_once()
        app_module._heartbeat_task = None

    @pytest.mark.asyncio
    @patch('src.api.app.start_periodic_heartbeat')
    @patch('src.api.app.initialize_connection_pool')
    @patch('src.api.app.initialize_database')
    @patch('src.api.app.close_connection_pool')
    @patch('src.api.app.flush_traces')
    @patch.dict('os.environ', {}, clear=False)
    async def test_lifespan_shutdown(
        self, mock_flush, mock_close, mock_init_db, mock_init_pool, mock_heartbeat
    ):
        """Test lifespan shutdown calls flush and close."""
        import os
        import src.api.app as app_module

        os.environ.pop('AWS_LAMBDA_FUNCTION_NAME', None)
        app_module._heartbeat_task = None

        mock_heartbeat.return_value = _make_mock_heartbeat_task()

        from src.api.app import lifespan

        app = Mock()

        async with lifespan(app):
            pass

        mock_flush.assert_called_once()
        mock_close.assert_called_once()
        app_module._heartbeat_task = None

    @pytest.mark.asyncio
    @patch('src.api.app.start_periodic_heartbeat')
    @patch('src.api.app.initialize_connection_pool')
    @patch('src.api.app.initialize_database')
    async def test_lifespan_startup_failure(
        self, mock_init_db, mock_init_pool, mock_heartbeat
    ):
        """Test lifespan startup failure propagates exception."""
        mock_init_pool.side_effect = Exception("Connection failed")

        from src.api.app import lifespan

        app = Mock()

        with pytest.raises(Exception, match="Connection failed"):
            async with lifespan(app):
                pass

    @pytest.mark.asyncio
    @patch('src.api.app.start_periodic_heartbeat')
    @patch('src.api.app.initialize_connection_pool')
    @patch('src.api.app.initialize_database')
    @patch('src.api.app.close_connection_pool')
    @patch('src.api.app.flush_traces')
    async def test_lifespan_skips_heartbeat_on_lambda(
        self, mock_flush, mock_close, mock_init_db, mock_init_pool, mock_heartbeat
    ):
        """Test heartbeat is not started when running on Lambda."""
        import os
        import src.api.app as app_module

        os.environ['AWS_LAMBDA_FUNCTION_NAME'] = 'my-lambda'
        app_module._heartbeat_task = None

        try:
            from src.api.app import lifespan

            app = Mock()

            async with lifespan(app):
                pass

            mock_heartbeat.assert_not_called()
            mock_flush.assert_called_once()
            mock_close.assert_called_once()
        finally:
            app_module._heartbeat_task = None
            os.environ.pop('AWS_LAMBDA_FUNCTION_NAME', None)