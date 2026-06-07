"""
Comprehensive tests for kafka.py module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.utils.kafka import (
    BaseKafkaLogger,
    KafkaLogger,
    KafkaEventLogger,
    ReasoningLogger,
    create_event_logger,
    create_reasoning_logger,
    kafka_logger
)


class TestBaseKafkaLogger:
    """Tests for BaseKafkaLogger class."""
    
    @patch('src.utils.kafka.KafkaManager')
    def test_initialization(self, mock_manager):
        """Test BaseKafkaLogger initialization."""
        logger = BaseKafkaLogger("test-topic", "TEST")
        
        assert logger.topic == "test-topic"
        assert logger.debug_prefix == "TEST"
    
    @patch('src.utils.kafka.KafkaManager')
    def test_on_send_success(self, mock_manager):
        """Test successful send callback."""
        logger = BaseKafkaLogger("test-topic", "TEST")
        
        mock_metadata = Mock()
        mock_metadata.topic = "test-topic"
        mock_metadata.partition = 0
        
        # Should not raise exception
        logger._on_send_success(mock_metadata)
    
    @patch('src.utils.kafka.KafkaManager')
    def test_on_send_error(self, mock_manager):
        """Test error send callback."""
        logger = BaseKafkaLogger("test-topic", "TEST")
        
        exception = Exception("Test error")
        
        # Should not raise exception
        logger._on_send_error(exception)
    
    @patch('src.utils.kafka.KafkaManager')
    def test_send_with_producer(self, mock_manager):
        """Test sending message with available producer."""
        mock_producer = Mock()
        mock_future = Mock()
        mock_producer.send.return_value = mock_future
        mock_manager.return_value.get_producer.return_value = mock_producer
        
        logger = BaseKafkaLogger("test-topic", "TEST")
        payload = {"key": "value"}
        
        logger._send(payload, show_debug=False)
        
        mock_producer.send.assert_called_once_with("test-topic", value=payload)
    
    @patch('src.utils.kafka.KafkaManager')
    def test_send_without_producer(self, mock_manager):
        """Test sending message without available producer."""
        mock_manager.return_value.get_producer.return_value = None
        
        logger = BaseKafkaLogger("test-topic", "TEST")
        payload = {"key": "value"}
        
        # Should not raise exception
        logger._send(payload, show_debug=False)
    
    @patch('src.utils.kafka.KafkaManager')
    def test_send_with_exception(self, mock_manager):
        """Test sending message with exception."""
        mock_producer = Mock()
        mock_producer.send.side_effect = Exception("Kafka error")
        mock_manager.return_value.get_producer.return_value = mock_producer
        
        logger = BaseKafkaLogger("test-topic", "TEST")
        payload = {"key": "value"}
        
        # Should not raise exception
        logger._send(payload, show_debug=False)


class TestKafkaLogger:
    """Tests for KafkaLogger class."""
    
    @patch.dict('os.environ', {'KAFKA_TOPIC_NAME': 'test-topic'})
    @patch('src.utils.kafka.KafkaManager')
    def test_initialization(self, mock_manager):
        """Test KafkaLogger initialization."""
        logger = KafkaLogger()
        
        assert logger.topic == 'test-topic'
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('src.utils.kafka.KafkaManager')
    def test_initialization_default_topic(self, mock_manager):
        """Test KafkaLogger with default topic."""
        logger = KafkaLogger()
        
        assert 'default' in logger.topic
    
    @patch('src.utils.kafka.KafkaManager')
    def test_log(self, mock_manager):
        """Test logging token usage data."""
        mock_producer = Mock()
        mock_manager.return_value.get_producer.return_value = mock_producer
        
        logger = KafkaLogger()
        data = {"tokens": 100, "model": "gpt-4"}
        
        logger.log(data)
        
        mock_producer.send.assert_called_once()
    
    @patch('src.utils.kafka.KafkaManager')
    def test_log_with_exception(self, mock_manager):
        """Test logging with exception."""
        mock_producer = Mock()
        mock_producer.send.side_effect = Exception("Kafka error")
        mock_manager.return_value.get_producer.return_value = mock_producer
        
        logger = KafkaLogger()
        data = {"tokens": 100}
        
        # Should not raise exception
        logger.log(data)
    
    @patch('src.utils.kafka.KafkaManager')
    def test_close(self, mock_manager):
        """Test closing Kafka logger."""
        mock_manager_instance = Mock()
        mock_manager.return_value = mock_manager_instance
        
        logger = KafkaLogger()
        logger.close()
        
        mock_manager_instance.close.assert_called_once()


class TestKafkaEventLogger:
    """Tests for KafkaEventLogger class."""
    
    @patch.dict('os.environ', {
        'KAFKA_EVENT_TOPIC_NAME': 'event-topic',
        'AGENT_NAME': 'TEST_AGENT',
        'SERVER_NAME': 'TEST_SERVER'
    })
    @patch('src.utils.kafka.KafkaManager')
    def test_initialization(self, mock_manager):
        """Test KafkaEventLogger initialization."""
        logger = KafkaEventLogger()
        
        assert logger.topic == 'event-topic'
        assert logger.agent_name == 'TEST_AGENT'
        assert logger.server_name == 'TEST_SERVER'
    
    @patch('src.utils.kafka.extract_user_context')
    @patch('src.utils.kafka.KafkaManager')
    def test_create_base_event(self, mock_manager, mock_extract):
        """Test creating base event."""
        mock_extract.return_value = {"encrypted_payload": "test_payload"}
        
        logger = KafkaEventLogger()
        event = logger._create_base_event("Test message", "auth_token")
        
        assert event['message'] == "Test message"
        assert event['encrypted_payload'] == "test_payload"
        assert 'timestamp' in event
        assert event['type'] == 'agent-event'
    
    @patch('src.utils.kafka.extract_user_context')
    @patch('src.utils.kafka.KafkaManager')
    def test_log_event(self, mock_manager, mock_extract):
        """Test logging event."""
        mock_extract.return_value = {"encrypted_payload": "test_payload"}
        mock_producer = Mock()
        mock_manager.return_value.get_producer.return_value = mock_producer
        
        logger = KafkaEventLogger()
        logger.log_event("Test event", "auth_token")
        
        mock_producer.send.assert_called_once()
    
    @patch('src.utils.kafka.extract_user_context')
    @patch('src.utils.kafka.KafkaManager')
    def test_log_event_no_payload(self, mock_manager, mock_extract):
        """Test logging event with NO_PAYLOAD."""
        mock_extract.return_value = {"encrypted_payload": "NO_PAYLOAD"}
        mock_producer = Mock()
        mock_manager.return_value.get_producer.return_value = mock_producer
        
        logger = KafkaEventLogger()
        logger.log_event("Test event")
        
        # Should still send but with show_debug=False
        mock_producer.send.assert_called_once()
    
    @patch('src.utils.kafka.KafkaManager')
    def test_close(self, mock_manager):
        """Test closing event logger."""
        mock_manager_instance = Mock()
        mock_manager.return_value = mock_manager_instance
        
        logger = KafkaEventLogger()
        logger.close()
        
        mock_manager_instance.close.assert_called_once()


class TestReasoningLogger:
    """Tests for ReasoningLogger class."""
    
    @patch.dict('os.environ', {
        'KAFKA_REASONING_TOPIC_NAME': 'reasoning-topic',
        'SERVER_NAME': 'TEST_SERVER'
    })
    @patch('src.utils.kafka.KafkaManager')
    def test_initialization(self, mock_manager):
        """Test ReasoningLogger initialization."""
        logger = ReasoningLogger()
        
        assert logger.topic == 'reasoning-topic'
        assert logger.server_name == 'TEST_SERVER'
    
    @patch('src.utils.kafka.extract_user_context')
    @patch('src.utils.kafka.KafkaManager')
    def test_log_reasoning(self, mock_manager, mock_extract):
        """Test logging reasoning."""
        mock_extract.return_value = {"encrypted_payload": "test_payload"}
        mock_producer = Mock()
        mock_manager.return_value.get_producer.return_value = mock_producer
        
        logger = ReasoningLogger()
        logger.log_reasoning("Test reasoning", "auth_token")
        
        mock_producer.send.assert_called_once()
    
    @patch('src.utils.kafka.extract_user_context')
    @patch('src.utils.kafka.KafkaManager')
    def test_log_reasoning_with_exception(self, mock_manager, mock_extract):
        """Test logging reasoning with exception."""
        mock_extract.side_effect = Exception("Extract error")
        
        logger = ReasoningLogger()
        
        # Should not raise exception
        logger.log_reasoning("Test reasoning")
    
    @patch('src.utils.kafka.KafkaManager')
    def test_close(self, mock_manager):
        """Test closing reasoning logger."""
        mock_manager_instance = Mock()
        mock_manager.return_value = mock_manager_instance
        
        logger = ReasoningLogger()
        logger.close()
        
        mock_manager_instance.close.assert_called_once()


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    @patch('src.utils.kafka.KafkaManager')
    def test_create_event_logger(self, mock_manager):
        """Test creating event logger."""
        logger = create_event_logger()
        
        assert isinstance(logger, KafkaEventLogger)
    
    @patch('src.utils.kafka.KafkaManager')
    def test_create_reasoning_logger(self, mock_manager):
        """Test creating reasoning logger."""
        logger = create_reasoning_logger()
        
        assert isinstance(logger, ReasoningLogger)
    
    @patch('src.utils.kafka.KafkaManager')
    def test_global_kafka_logger(self, mock_manager):
        """Test global kafka_logger instance."""
        assert kafka_logger is not None
        assert isinstance(kafka_logger, KafkaLogger)
