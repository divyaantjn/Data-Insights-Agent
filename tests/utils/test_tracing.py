"""
Comprehensive tests for tracing.py module.
"""
import sys
from unittest.mock import MagicMock

sys.modules["opentelemetry"] = MagicMock()
sys.modules["opentelemetry.sdk"] = MagicMock()
sys.modules["opentelemetry.sdk.trace"] = MagicMock()
sys.modules["opentelemetry.sdk.resources"] = MagicMock()
sys.modules["opentelemetry.sdk.trace.export"] = MagicMock()
sys.modules["opentelemetry.instrumentation"] = MagicMock()
sys.modules["opentelemetry.instrumentation.fastapi"] = MagicMock()
sys.modules["opentelemetry.instrumentation.requests"] = MagicMock()
sys.modules["opentelemetry.instrumentation.httpx"] = MagicMock()
sys.modules["opentelemetry.exporter"] = MagicMock()
sys.modules["opentelemetry.exporter.otlp"] = MagicMock()
sys.modules["opentelemetry.exporter.otlp.proto"] = MagicMock()
sys.modules["opentelemetry.exporter.otlp.proto.http"] = MagicMock()
sys.modules["opentelemetry.exporter.otlp.proto.http.trace_exporter"] = MagicMock()

import src.utils.tracing
import pytest
import os
from unittest.mock import Mock, patch, MagicMock, call


class TestGetTracerProvider:
    """Tests for get_tracer_provider function."""
    
    @patch('src.utils.tracing._tracer_provider', None)
    @patch('src.utils.tracing.trace.get_tracer_provider')
    def test_get_tracer_provider_none(self, mock_get_provider):
        """Test getting tracer provider when none is set."""
        import src.utils.tracing as tracing_module
        with patch.object(tracing_module, '_tracer_provider', None), \
             patch('src.utils.tracing.trace.get_tracer_provider') as mock_get_provider:
            from src.utils.tracing import get_tracer_provider
            mock_provider = Mock()
            mock_get_provider.return_value = mock_provider
            result = get_tracer_provider()
            assert result == mock_provider
            mock_get_provider.assert_called_once()
    
    @patch('src.utils.tracing._tracer_provider')
    def test_get_tracer_provider_exists(self, mock_tracer_provider):
        """Test getting tracer provider when one exists."""
        import src.utils.tracing as tracing_module
        mock_provider = Mock()
        with patch.object(tracing_module, '_tracer_provider', mock_provider):
            from src.utils.tracing import get_tracer_provider
            result = get_tracer_provider()
            assert result == mock_provider


class TestSetupTracing:
    """Tests for setup_tracing function."""
    
    @patch.dict('os.environ', {'OTEL_TRACING_ENABLED': 'false'})
    def test_setup_tracing_disabled(self):
        """Test setup when tracing is disabled."""
        from src.utils.tracing import setup_tracing
        
        result = setup_tracing()
        
        assert result is False
    
    @patch.dict('os.environ', {
        'OTEL_TRACING_ENABLED': 'true',
        'OTEL_SERVICE_NAME': 'test-service',
        'OTEL_ENVIRONMENT': 'test',
        'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4318/v1/traces',
        'OTEL_EXPORTER_OTLP_TIMEOUT': '15'
    })
    @patch('src.utils.tracing.Resource')
    @patch('src.utils.tracing.OTLPSpanExporter')
    @patch('src.utils.tracing.BatchSpanProcessor')
    @patch('src.utils.tracing.TracerProvider')
    @patch('src.utils.tracing.trace.set_tracer_provider')
    @patch('src.utils.tracing.HTTPXClientInstrumentor')
    @patch('src.utils.tracing.RequestsInstrumentor')
    def test_setup_tracing_success(
        self, mock_requests_inst, mock_httpx_inst, mock_set_provider,
        mock_tracer_provider_class, mock_batch_processor, mock_exporter, mock_resource
    ):
        """Test successful tracing setup."""
        from src.utils.tracing import setup_tracing
        
        # Setup mocks
        mock_resource_instance = Mock()
        mock_resource.create.return_value = mock_resource_instance
        
        mock_exporter_instance = Mock()
        mock_exporter.return_value = mock_exporter_instance
        
        mock_processor_instance = Mock()
        mock_batch_processor.return_value = mock_processor_instance
        
        mock_provider_instance = Mock()
        mock_tracer_provider_class.return_value = mock_provider_instance
        
        mock_httpx_instrumentor = Mock()
        mock_httpx_inst.return_value = mock_httpx_instrumentor
        
        mock_requests_instrumentor = Mock()
        mock_requests_inst.return_value = mock_requests_instrumentor
        
        result = setup_tracing()
        
        assert result is True
        mock_resource.create.assert_called_once()
        mock_exporter.assert_called_once_with(
            endpoint='http://localhost:4318/v1/traces',
            timeout=15
        )
        mock_batch_processor.assert_called_once_with(mock_exporter_instance)
        mock_provider_instance.add_span_processor.assert_called_once_with(mock_processor_instance)
        mock_set_provider.assert_called_once_with(mock_provider_instance)
        mock_httpx_instrumentor.instrument.assert_called_once()
        mock_requests_instrumentor.instrument.assert_called_once()
    
    @patch.dict('os.environ', {'OTEL_TRACING_ENABLED': 'true'}, clear=True)
    def test_setup_tracing_no_endpoint(self):
        """Test setup when OTLP endpoint is not set."""
        from src.utils.tracing import setup_tracing
        
        result = setup_tracing()
        
        assert result is False
    
    @patch.dict('os.environ', {
        'OTEL_TRACING_ENABLED': 'true',
        'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4318/v1/traces'
    })
    @patch('src.utils.tracing.Resource')
    def test_setup_tracing_with_defaults(self, mock_resource):
        """Test setup with default service name and environment."""
        from src.utils.tracing import setup_tracing
        
        mock_resource.create.return_value = Mock()
        
        with patch('src.utils.tracing.OTLPSpanExporter'), \
             patch('src.utils.tracing.BatchSpanProcessor'), \
             patch('src.utils.tracing.TracerProvider'), \
             patch('src.utils.tracing.trace.set_tracer_provider'), \
             patch('src.utils.tracing.HTTPXClientInstrumentor'), \
             patch('src.utils.tracing.RequestsInstrumentor'):
            
            result = setup_tracing()
            
            assert result is True
            # Verify default values were used
            call_args = mock_resource.create.call_args[0][0]
            assert call_args['service.name'] == 'data-insights-backend'
            assert call_args['service.environment'] == 'dev'
    
    @patch.dict('os.environ', {
        'OTEL_TRACING_ENABLED': 'true',
        'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4318/v1/traces'
    })
    @patch('src.utils.tracing.Resource')
    def test_setup_tracing_with_custom_params(self, mock_resource):
        """Test setup with custom parameters."""
        from src.utils.tracing import setup_tracing
        
        mock_resource.create.return_value = Mock()
        
        with patch('src.utils.tracing.OTLPSpanExporter') as mock_exporter, \
             patch('src.utils.tracing.BatchSpanProcessor'), \
             patch('src.utils.tracing.TracerProvider'), \
             patch('src.utils.tracing.trace.set_tracer_provider'), \
             patch('src.utils.tracing.HTTPXClientInstrumentor'), \
             patch('src.utils.tracing.RequestsInstrumentor'):
            
            result = setup_tracing(
                service_name='custom-service',
                service_environment='production',
                otlp_endpoint='http://custom:4318/v1/traces',
                otlp_timeout=30
            )
            
            assert result is True
            # Verify custom values were used
            call_args = mock_resource.create.call_args[0][0]
            assert call_args['service.name'] == 'custom-service'
            assert call_args['service.environment'] == 'production'
            
            exporter_call_args = mock_exporter.call_args[1]
            assert exporter_call_args['endpoint'] == 'http://custom:4318/v1/traces'
            assert exporter_call_args['timeout'] == 30
    
    @patch.dict('os.environ', {
        'OTEL_TRACING_ENABLED': 'true',
        'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4318/v1/traces'
    })
    @patch('src.utils.tracing.Resource')
    def test_setup_tracing_exception(self, mock_resource):
        """Test setup with exception during initialization."""
        from src.utils.tracing import setup_tracing
        
        mock_resource.create.side_effect = Exception("Setup failed")
        
        result = setup_tracing()
        
        assert result is False
    
    @patch.dict('os.environ', {
        'OTEL_TRACING_ENABLED': 'TRUE',  # Test case insensitivity
        'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4318/v1/traces'
    })
    @patch('src.utils.tracing.Resource')
    def test_setup_tracing_enabled_case_insensitive(self, mock_resource):
        """Test that OTEL_TRACING_ENABLED is case insensitive."""
        from src.utils.tracing import setup_tracing
        
        mock_resource.create.return_value = Mock()
        
        with patch('src.utils.tracing.OTLPSpanExporter'), \
             patch('src.utils.tracing.BatchSpanProcessor'), \
             patch('src.utils.tracing.TracerProvider'), \
             patch('src.utils.tracing.trace.set_tracer_provider'), \
             patch('src.utils.tracing.HTTPXClientInstrumentor'), \
             patch('src.utils.tracing.RequestsInstrumentor'):
            
            result = setup_tracing()
            
            assert result is True


class TestInstrumentFastAPIApp:
    """Tests for instrument_fastapi_app function."""
    
    @patch('src.utils.tracing.get_tracer_provider')
    @patch('src.utils.tracing.FastAPIInstrumentor')
    def test_instrument_fastapi_app_success(self, mock_instrumentor, mock_get_provider):
        """Test successful FastAPI app instrumentation."""
        from src.utils.tracing import instrument_fastapi_app
        
        mock_app = Mock()
        mock_provider = Mock()
        mock_resource = Mock()
        mock_resource.attributes = {'service.name': 'test-service'}
        mock_provider.resource = mock_resource
        mock_get_provider.return_value = mock_provider
        
        instrument_fastapi_app(mock_app)
        
        mock_instrumentor.instrument_app.assert_called_once_with(
            mock_app,
            tracer_provider=mock_provider
        )
    
    @patch('src.utils.tracing.get_tracer_provider')
    def test_instrument_fastapi_app_no_provider(self, mock_get_provider):
        """Test FastAPI instrumentation when no provider exists."""
        from src.utils.tracing import instrument_fastapi_app
        
        mock_app = Mock()
        mock_get_provider.return_value = None
        
        # Should not raise exception
        instrument_fastapi_app(mock_app)
    
    @patch('src.utils.tracing.get_tracer_provider')
    def test_instrument_fastapi_app_no_resource(self, mock_get_provider):
        """Test FastAPI instrumentation when provider has no resource."""
        from src.utils.tracing import instrument_fastapi_app
        
        mock_app = Mock()
        mock_provider = Mock(spec=[])  # No resource attribute
        mock_get_provider.return_value = mock_provider
        
        # Should not raise exception
        instrument_fastapi_app(mock_app)
    
    @patch('src.utils.tracing.get_tracer_provider')
    @patch('src.utils.tracing.FastAPIInstrumentor')
    def test_instrument_fastapi_app_exception(self, mock_instrumentor, mock_get_provider):
        """Test FastAPI instrumentation with exception."""
        from src.utils.tracing import instrument_fastapi_app
        
        mock_app = Mock()
        mock_provider = Mock()
        mock_resource = Mock()
        mock_resource.attributes = {'service.name': 'test-service'}
        mock_provider.resource = mock_resource
        mock_get_provider.return_value = mock_provider
        
        mock_instrumentor.instrument_app.side_effect = Exception("Instrumentation failed")
        
        # Should not raise exception
        instrument_fastapi_app(mock_app)


class TestGetTracer:
    """Tests for get_tracer function."""
    
    @patch('src.utils.tracing.trace.get_tracer')
    def test_get_tracer_default_name(self, mock_get_tracer):
        """Test getting tracer with default name."""
        from src.utils.tracing import get_tracer
        
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        
        result = get_tracer()
        
        assert result == mock_tracer
        mock_get_tracer.assert_called_once()
    
    @patch('src.utils.tracing.trace.get_tracer')
    def test_get_tracer_custom_name(self, mock_get_tracer):
        """Test getting tracer with custom name."""
        from src.utils.tracing import get_tracer
        
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        
        result = get_tracer('custom.tracer')
        
        assert result == mock_tracer
        mock_get_tracer.assert_called_once_with('custom.tracer')


class TestFlushTraces:
    """Tests for flush_traces function."""
    
    @patch('src.utils.tracing.get_tracer_provider')
    def test_flush_traces_success(self, mock_get_provider):
        """Test successful trace flushing."""
        from src.utils.tracing import flush_traces
        
        mock_provider = Mock()
        mock_provider.force_flush.return_value = True
        mock_get_provider.return_value = mock_provider
        
        result = flush_traces()
        
        assert result is True
        mock_provider.force_flush.assert_called_once_with(5000)
    
    @patch('src.utils.tracing.get_tracer_provider')
    def test_flush_traces_with_custom_timeout(self, mock_get_provider):
        """Test trace flushing with custom timeout."""
        from src.utils.tracing import flush_traces
        
        mock_provider = Mock()
        mock_provider.force_flush.return_value = True
        mock_get_provider.return_value = mock_provider
        
        result = flush_traces(timeout_millis=10000)
        
        assert result is True
        mock_provider.force_flush.assert_called_once_with(10000)
    
    @patch('src.utils.tracing.get_tracer_provider')
    def test_flush_traces_timeout(self, mock_get_provider):
        """Test trace flushing with timeout."""
        from src.utils.tracing import flush_traces
        
        mock_provider = Mock()
        mock_provider.force_flush.return_value = False
        mock_get_provider.return_value = mock_provider
        
        result = flush_traces()
        
        assert result is False
    
    @patch('src.utils.tracing.get_tracer_provider')
    def test_flush_traces_no_provider(self, mock_get_provider):
        """Test trace flushing when no provider exists."""
        from src.utils.tracing import flush_traces
        
        mock_get_provider.return_value = None
        
        result = flush_traces()
        
        assert result is False
    
    @patch('src.utils.tracing.get_tracer_provider')
    def test_flush_traces_no_force_flush(self, mock_get_provider):
        """Test trace flushing when provider doesn't support force_flush."""
        from src.utils.tracing import flush_traces
        
        mock_provider = Mock(spec=[])  # No force_flush method
        mock_get_provider.return_value = mock_provider
        
        result = flush_traces()
        
        assert result is False
    
    @patch('src.utils.tracing.get_tracer_provider')
    def test_flush_traces_exception(self, mock_get_provider):
        """Test trace flushing with exception."""
        from src.utils.tracing import flush_traces
        
        mock_provider = Mock()
        mock_provider.force_flush.side_effect = Exception("Flush failed")
        mock_get_provider.return_value = mock_provider
        
        result = flush_traces()
        
        assert result is False


class TestTracingIntegration:
    """Integration tests for tracing module."""
    
    @patch.dict('os.environ', {
        'OTEL_TRACING_ENABLED': 'true',
        'OTEL_SERVICE_NAME': 'integration-test',
        'OTEL_ENVIRONMENT': 'test',
        'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4318/v1/traces'
    })
    @patch('src.utils.tracing.Resource')
    @patch('src.utils.tracing.OTLPSpanExporter')
    @patch('src.utils.tracing.BatchSpanProcessor')
    @patch('src.utils.tracing.TracerProvider')
    @patch('src.utils.tracing.trace.set_tracer_provider')
    @patch('src.utils.tracing.HTTPXClientInstrumentor')
    @patch('src.utils.tracing.RequestsInstrumentor')
    def test_full_tracing_lifecycle(
        self, mock_requests_inst, mock_httpx_inst, mock_set_provider,
        mock_tracer_provider_class, mock_batch_processor, mock_exporter, mock_resource
    ):
        """Test full tracing lifecycle: setup, instrument, flush."""
        from src.utils.tracing import setup_tracing, instrument_fastapi_app, flush_traces, get_tracer_provider
        
        # Setup mocks
        mock_resource_instance = Mock()
        mock_resource.create.return_value = mock_resource_instance
        
        mock_provider_instance = Mock()
        mock_provider_instance.resource = mock_resource_instance
        mock_provider_instance.force_flush.return_value = True
        mock_tracer_provider_class.return_value = mock_provider_instance
        
        mock_httpx_instrumentor = Mock()
        mock_httpx_inst.return_value = mock_httpx_instrumentor
        
        mock_requests_instrumentor = Mock()
        mock_requests_inst.return_value = mock_requests_instrumentor
        
        # Setup tracing
        setup_result = setup_tracing()
        assert setup_result is True
        
        # Get provider
        provider = get_tracer_provider()
        assert provider == mock_provider_instance
        
        # Instrument FastAPI app
        mock_app = Mock()
        with patch('src.utils.tracing.FastAPIInstrumentor') as mock_fastapi_inst:
            instrument_fastapi_app(mock_app)
            mock_fastapi_inst.instrument_app.assert_called_once()
        
        # Flush traces
        flush_result = flush_traces()
        assert flush_result is True


class TestModuleGlobals:
    """Tests for module-level globals."""
    
    def test_tracer_provider_global_exists(self):
        """Test that _tracer_provider global exists."""
        import src.utils.tracing as tracing_module
        
        assert hasattr(tracing_module, '_tracer_provider')
    
    def test_logger_exists(self):
        """Test that logger exists."""
        import src.utils.tracing as tracing_module
        
        assert hasattr(tracing_module, 'logger')
        assert tracing_module.logger is not None
