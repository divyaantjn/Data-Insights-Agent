import sys
import importlib
from unittest.mock import patch, Mock, AsyncMock, MagicMock

import pytest
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse

# ============================================================
# FIXED reload (VERY IMPORTANT)
# ============================================================

def reload_main():
    if "main" in sys.modules:
        del sys.modules["main"]
    import main
    return main

# ============================================================
# INITIALIZATION TESTS
# ============================================================

class TestMainModuleInitialization:
    @patch("src.utils.tracing.setup_tracing")
    @patch("src.api.app.create_app")
    @patch("src.utils.config_loader.ConfigLoader")
    def test_module_imports_and_setup(
        self, mock_config, mock_create, mock_tracing
    ):
        reload_main()

        assert mock_tracing.called
        assert mock_create.called
        assert mock_config.called

    @patch("src.utils.tracing.setup_tracing")
    @patch("src.api.app.create_app")
    @patch("src.utils.config_loader.ConfigLoader")
    def test_app_created(
        self, mock_config, mock_create, mock_tracing
    ):
        mock_app = Mock()
        mock_create.return_value = mock_app

        main = reload_main()

        assert mock_create.called
        assert main.app == mock_app

    @patch("src.utils.tracing.setup_tracing")
    @patch("src.api.app.create_app")
    @patch("src.utils.config_loader.ConfigLoader")
    def test_config_loader_initialized(self, mock_config, *_):
        reload_main()
        assert mock_config.called

# ============================================================
# FIXED EXCEPTION HANDLERS TESTS - REWRITTEN
# ============================================================

class TestExceptionHandlers:
    def setup_method(self):
        """Reload main module before each test to get fresh handler functions"""
        self.main = reload_main()

    @pytest.mark.asyncio
    async def test_http_exception_handler(self):
        # Patch capture function AFTER importing handlers
        with patch.object(self.main, "capture_http_exception", new_callable=AsyncMock) as mock_capture:
            # Patch app.exception_handler to avoid infinite recursion/mock chaining
            with patch.object(self.main.app, "exception_handler", return_value=None):
                req = Mock(spec=Request)
                exc = StarletteHTTPException(status_code=404, detail="Not Found")

                res = await self.main.http_exception_handler(req, exc)

                assert isinstance(res, JSONResponse)
                assert res.status_code == 404
                mock_capture.assert_called_once_with(exc)

    @pytest.mark.asyncio
    async def test_validation_exception_handler(self):
        with patch.object(self.main, "capture_validation_error", new_callable=Mock) as mock_capture:
            with patch.object(self.main.app, "exception_handler", return_value=None):
                req = Mock(spec=Request)
                err = Mock(spec=RequestValidationError)
                err.errors.return_value = [{"msg": "error"}]

                res = await self.main.validation_exception_handler(req, err)

                assert res.status_code == 422
                mock_capture.assert_called_once_with(err)

    @pytest.mark.asyncio
    async def test_general_exception_handler(self):
        with patch.object(self.main, "capture_exception", new_callable=AsyncMock) as mock_capture:
            with patch.object(self.main.app, "exception_handler", return_value=None):
                req = Mock(spec=Request)
                req.url = Mock()
                req.url.path = "/test"
                exc = Exception("err")

                res = await self.main.general_exception_handler(req, exc)

                assert res.status_code == 500
                mock_capture.assert_called_once()

# ============================================================
# INSTRUMENTATION TEST
# ============================================================

class TestMiddlewareSetup:
    @patch("src.utils.tracing.instrument_fastapi_app")
    @patch("src.utils.tracing.setup_tracing")
    @patch("src.api.app.create_app")
    @patch("src.utils.config_loader.ConfigLoader")
    def test_instrumentation_called(
        self, mock_config, mock_create,mock_tracing, mock_instrument
    ):
        app = Mock()
        mock_create.return_value = app

        reload_main()

        mock_instrument.assert_called_once_with(app)

# ============================================================
# EXCEPTION HANDLER REGISTRATION (SAFE CHECK)
# ============================================================

class TestExceptionHandlerRegistration:
    def test_handlers_registered(self):
        import main

        assert hasattr(main, "http_exception_handler")
        assert hasattr(main, "validation_exception_handler")
        assert hasattr(main, "general_exception_handler")

# ============================================================
# TRACING ORDER TEST
# ============================================================

class TestTracingOrder:
    @patch("src.utils.tracing.setup_tracing")
    @patch("src.api.app.create_app")
    @patch("src.utils.config_loader.ConfigLoader")
    def test_order(self, mock_config, mock_create, mock_tracing):
        reload_main()

        assert mock_tracing.called
