"""
Tests for error_capture.py module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.utils.error_capture import (
    capture_http_exception,
    capture_validation_error,
    capture_exception,
    capture_http_error_details,
    capture_validation_error_list,
    capture_external_api_error
)


class TestCaptureHttpException:
    """Tests for capture_http_exception function."""
    
    def test_capture_http_exception_4xx(self, mock_xray_recorder, mock_otel_span):
        """Test capturing 4xx HTTP exception."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            exc = Mock()
            exc.status_code = 404
            exc.detail = "Not found"
            
            capture_http_exception(exc)
            
            # Verify X-Ray annotations
            assert mock_set_xray.call_count >= 5
            mock_set_xray.assert_any_call("error_occurred", True)
            mock_set_xray.assert_any_call("error_status_code", 404)
            mock_set_xray.assert_any_call("error_category", "client_error")
            mock_set_xray.assert_any_call("error_severity", "warning")
            
            # Verify OTEL span attributes
            mock_otel_span.set_attribute.assert_any_call("error", True)
            mock_otel_span.set_attribute.assert_any_call("error.status_code", 404)
    
    def test_capture_http_exception_5xx(self, mock_xray_recorder, mock_otel_span):
        """Test capturing 5xx HTTP exception."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            exc = Mock()
            exc.status_code = 500
            exc.detail = "Internal server error"
            
            capture_http_exception(exc)
            
            mock_set_xray.assert_any_call("error_category", "server_error")
            mock_set_xray.assert_any_call("error_severity", "critical")
    
    def test_capture_http_exception_no_span(self, mock_xray_recorder):
        """Test capturing exception when no OTEL span is available."""
        with patch('src.utils.error_capture.set_xray_annotation'), \
             patch('src.utils.error_capture.trace.get_current_span', return_value=None):
            
            exc = Mock()
            exc.status_code = 400
            exc.detail = "Bad request"
            
            # Should not raise exception
            capture_http_exception(exc)
    
    def test_capture_http_exception_with_error(self, mock_xray_recorder):
        """Test capturing exception when annotation fails."""
        with patch('src.utils.error_capture.set_xray_annotation', side_effect=Exception("Test error")), \
             patch('src.utils.error_capture.trace.get_current_span', return_value=None):
            
            exc = Mock()
            exc.status_code = 400
            exc.detail = "Bad request"
            
            # Should not raise exception
            capture_http_exception(exc)


class TestCaptureValidationError:
    """Tests for capture_validation_error function."""
    
    def test_capture_validation_error_with_errors_method(self, mock_xray_recorder, mock_otel_span):
        """Test capturing validation error with errors() method."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            exc = Mock()
            exc.errors = Mock(return_value=[
                {
                    'loc': ('body', 'field_name'),
                    'type': 'value_error',
                    'msg': 'Invalid value'
                }
            ])
            
            capture_validation_error(exc)
            
            mock_set_xray.assert_any_call("error_occurred", True)
            mock_set_xray.assert_any_call("error_type", "validation_error")
            mock_set_xray.assert_any_call("error_count", 1)
            mock_set_xray.assert_any_call("error_field", "field_name")
    
    def test_capture_validation_error_with_errors_attribute(self, mock_xray_recorder, mock_otel_span):
        """Test capturing validation error with errors attribute."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            exc = Mock()
            exc.errors = [
                {
                    'loc': ('query', 'param'),
                    'type': 'type_error',
                    'msg': 'Type error'
                }
            ]
            
            capture_validation_error(exc)
            
            mock_set_xray.assert_any_call("error_field", "param")
            mock_set_xray.assert_any_call("validation_error_type", "type_error")
    
    def test_capture_validation_error_with_args(self, mock_xray_recorder, mock_otel_span):
        """Test capturing validation error with args."""
        with patch('src.utils.error_capture.set_xray_annotation'), \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            exc = Mock()
            exc.errors = None
            exc.args = ("Validation failed",)
            
            capture_validation_error(exc)
    
    def test_capture_validation_error_empty_errors(self, mock_xray_recorder, mock_otel_span):
        """Test capturing validation error with empty errors list."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            exc = Mock()
            exc.errors = Mock(return_value=[])
            
            capture_validation_error(exc)
            
            mock_set_xray.assert_any_call("error_count", 0)


class TestCaptureException:
    """Tests for capture_exception function."""
    
    def test_capture_exception_basic(self, mock_xray_recorder, mock_otel_span):
        """Test capturing basic exception."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            exc = ValueError("Test error")
            
            capture_exception(exc)
            
            mock_set_xray.assert_any_call("error_occurred", True)
            mock_set_xray.assert_any_call("error_type", "ValueError")
            mock_set_xray.assert_any_call("error_category", "unhandled_exception")
            mock_otel_span.record_exception.assert_called_once_with(exc)
    
    def test_capture_exception_with_context(self, mock_xray_recorder, mock_otel_span):
        """Test capturing exception with context."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            exc = RuntimeError("Runtime error")
            context = {
                'user_id': 'user123',
                'operation': 'data_processing',
                'count': 42
            }
            
            capture_exception(exc, context)
            
            mock_set_xray.assert_any_call("error_context_user_id", "user123")
            mock_set_xray.assert_any_call("error_context_operation", "data_processing")
            mock_otel_span.set_attribute.assert_any_call("error.context.user_id", "user123")
    
    def test_capture_exception_long_message(self, mock_xray_recorder, mock_otel_span):
        """Test capturing exception with long message."""
        with patch('src.utils.error_capture.set_xray_annotation'), \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            long_message = "A" * 1000
            exc = Exception(long_message)
            
            capture_exception(exc)
            
            # Should truncate message
            mock_otel_span.set_attribute.assert_called()


class TestCaptureHttpErrorDetails:
    """Tests for capture_http_error_details function."""
    
    def test_capture_http_error_details_basic(self, mock_xray_recorder, mock_otel_span):
        """Test capturing HTTP error details."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            error_details = {
                'message': 'Resource not found',
                'error_key': 'NOT_FOUND'
            }
            
            capture_http_error_details(404, error_details)
            
            mock_set_xray.assert_any_call("error_occurred", True)
            mock_set_xray.assert_any_call("error_status_code", 404)
            mock_set_xray.assert_any_call("error_message", "Resource not found")
            mock_set_xray.assert_any_call("error_key", "NOT_FOUND")
    
    def test_capture_http_error_details_with_validation_details(self, mock_xray_recorder, mock_otel_span):
        """Test capturing HTTP error with validation details."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            error_details = {
                'message': 'Validation failed',
                'details': [
                    {
                        'loc': ('body', 'email'),
                        'type': 'value_error.email'
                    }
                ]
            }
            
            capture_http_error_details(422, error_details)
            
            mock_set_xray.assert_any_call("error_field", "email")
            mock_set_xray.assert_any_call("error_type", "value_error.email")
    
    def test_capture_http_error_details_no_details(self, mock_xray_recorder, mock_otel_span):
        """Test capturing HTTP error without details."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            capture_http_error_details(500)
            
            mock_set_xray.assert_any_call("error_category", "server_error")
            mock_set_xray.assert_any_call("error_severity", "critical")


class TestCaptureValidationErrorList:
    """Tests for capture_validation_error_list function."""
    
    def test_capture_validation_error_list(self, mock_xray_recorder, mock_otel_span):
        """Test capturing validation error list."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            errors = [
                {
                    'loc': ('body', 'username'),
                    'type': 'value_error.missing',
                    'msg': 'Field required'
                },
                {
                    'loc': ('body', 'password'),
                    'type': 'value_error.any_str.min_length',
                    'msg': 'Too short'
                }
            ]
            
            capture_validation_error_list(errors)
            
            mock_set_xray.assert_any_call("error_count", 2)
            mock_set_xray.assert_any_call("error_field", "username")
            mock_set_xray.assert_any_call("validation_error_type", "value_error.missing")


class TestCaptureExternalApiError:
    """Tests for capture_external_api_error function."""
    
    def test_capture_external_api_error_complete(self, mock_xray_recorder, mock_otel_span):
        """Test capturing external API error with all parameters."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            capture_external_api_error(
                service_name="payment-service",
                endpoint="/api/v1/charge",
                status_code=503,
                error_message="Service unavailable",
                error_type="timeout"
            )
            
            mock_set_xray.assert_any_call("external_api_error", True)
            mock_set_xray.assert_any_call("external_service", "payment-service")
            mock_set_xray.assert_any_call("external_status_code", 503)
            mock_set_xray.assert_any_call("external_error_type", "timeout")
            
            mock_otel_span.set_attribute.assert_any_call("external.api.error", True)
            mock_otel_span.set_attribute.assert_any_call("external.service.name", "payment-service")
    
    def test_capture_external_api_error_minimal(self, mock_xray_recorder, mock_otel_span):
        """Test capturing external API error with minimal parameters."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            capture_external_api_error(
                service_name="auth-service",
                endpoint="/validate"
            )
            
            mock_set_xray.assert_any_call("external_api_error", True)
            mock_set_xray.assert_any_call("external_service", "auth-service")
    
    def test_capture_external_api_error_long_endpoint(self, mock_xray_recorder, mock_otel_span):
        """Test capturing external API error with long endpoint."""
        with patch('src.utils.error_capture.set_xray_annotation') as mock_set_xray, \
             patch('src.utils.error_capture.trace.get_current_span', return_value=mock_otel_span):
            
            long_endpoint = "/api/v1/" + "a" * 300
            
            capture_external_api_error(
                service_name="test-service",
                endpoint=long_endpoint
            )
            
            # Should truncate endpoint
            mock_set_xray.assert_called()
