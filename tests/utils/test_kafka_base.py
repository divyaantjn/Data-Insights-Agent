"""
Comprehensive tests for kafka_base.py module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.utils.kafka_base import (
    extract_user_context,
    KafkaManager,
    CUSTOM_TOKEN_SEPARATOR,
    KAFKA_INSTALLED
)


class TestExtractUserContext:
    """Tests for extract_user_context function."""
    
    def test_extract_user_context_no_token(self):
        """Test extracting context without token."""
        result = extract_user_context(None)
        
        assert result["encrypted_payload"] == "NO_PAYLOAD"
    
    def test_extract_user_context_with_separator(self):
        """Test extracting context with custom separator."""
        token = f"jwt_part{CUSTOM_TOKEN_SEPARATOR}encrypted_data_here"
        
        result = extract_user_context(token)
        
        assert result["encrypted_payload"] == "encrypted_data_here"
    
    def test_extract_user_context_without_separator(self):
        """Test extracting context without separator."""
        token = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test_token_data"
        
        result = extract_user_context(token)
        
        assert "mock-encrypted-payload" in result["encrypted_payload"]
        assert result["encrypted_payload"] != "NO_PAYLOAD"
    
    def test_extract_user_context_bearer_prefix(self):
        """Test extracting context with Bearer prefix."""
        token = "bearer test_token_12345"
        
        result = extract_user_context(token)
        
        assert "mock-encrypted-payload" in result["encrypted_payload"]
    
    def test_extract_user_context_empty_string(self):
        """Test extracting context with empty string."""
        result = extract_user_context("")
        
        assert result["encrypted_payload"] == "NO_PAYLOAD"
    
    def test_extract_user_context_malformed_token(self):
        """Test extracting context with malformed token."""
        token = "malformed"
        
        result = extract_user_context(token)
        
        # Should handle gracefully
        assert "encrypted_payload" in result


class TestKafkaManagerSingleton:
    """Tests for KafkaManager singleton pattern."""
    
    def test_singleton_instance(self):
        """Test that KafkaManager is a singleton."""
        manager1 = KafkaManager()
        manager2 = KafkaManager()
        
        assert manager1 is manager2
    
    def test_initialization_once(self):
        """Test that initialization happens only once."""
        manager = KafkaManager()
        
        assert manager._initialized is True
        assert manager.producer is None
    
    def test_multiple_instantiation(self):
        """Test multiple instantiation returns same instance."""
        managers = [KafkaManager() for _ in range(5)]
        
        # All should be the same instance
        assert all(m is managers[0] for m in managers)


class TestKafkaManagerProducer:
    """Tests for KafkaManager producer management."""
    
    @patch.dict('os.environ', {'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092'})
    @patch('src.utils.kafka_base.KAFKA_INSTALLED', True)
    @patch('src.utils.kafka_base.KafkaProducer')
    def test_get_producer_initializes_once(self, mock_producer_class):
        """Test that producer is initialized only once."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer
        
        # Reset singleton for test
        KafkaManager._instance = None
        manager = KafkaManager()
        
        producer1 = manager.get_producer()
        producer2 = manager.get_producer()
        
        assert producer1 is producer2
        mock_producer_class.assert_called_once()
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('src.utils.kafka_base.KAFKA_INSTALLED', True)
    def test_get_producer_no_bootstrap_servers(self):
        """Test get_producer without bootstrap servers."""
        # Reset singleton for test
        KafkaManager._instance = None
        manager = KafkaManager()
        
        producer = manager.get_producer()
        
        assert producer is None
    
    @patch('src.utils.kafka_base.KAFKA_INSTALLED', False)
    def test_get_producer_kafka_not_installed(self):
        """Test get_producer when Kafka is not installed."""
        # Reset singleton for test
        KafkaManager._instance = None
        manager = KafkaManager()
        
        producer = manager.get_producer()
        
        assert producer is None
    
    @patch.dict('os.environ', {'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092'})
    @patch('src.utils.kafka_base.KAFKA_INSTALLED', True)
    @patch('src.utils.kafka_base.KafkaProducer')
    def test_initialize_producer_success(self, mock_producer_class):
        """Test successful producer initialization."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer
        
        # Reset singleton for test
        KafkaManager._instance = None
        manager = KafkaManager()
        
        result = manager._initialize_producer()
        
        assert result is not False
        assert manager.producer is not None
    
    @patch.dict('os.environ', {'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092'})
    @patch('src.utils.kafka_base.KAFKA_INSTALLED', True)
    @patch('src.utils.kafka_base.KafkaProducer')
    def test_initialize_producer_connection_error(self, mock_producer_class):
        """Test producer initialization with connection error."""
        from src.utils.kafka_base import NoBrokersAvailable
        mock_producer_class.side_effect = NoBrokersAvailable("No brokers")
        
        # Reset singleton for test
        KafkaManager._instance = None
        manager = KafkaManager()
        
        result = manager._initialize_producer()
        
        assert result is False
        assert manager.producer is None
    
    @patch.dict('os.environ', {'KAFKA_BOOTSTRAP_SERVERS': 'server1:9092,server2:9092'})
    @patch('src.utils.kafka_base.KAFKA_INSTALLED', True)
    @patch('src.utils.kafka_base.KafkaProducer')
    def test_initialize_producer_multiple_servers(self, mock_producer_class):
        """Test producer initialization with multiple bootstrap servers."""
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer
        
        # Reset singleton for test
        KafkaManager._instance = None
        manager = KafkaManager()
        
        manager._initialize_producer()
        
        # Check that bootstrap_servers was split correctly
        call_kwargs = mock_producer_class.call_args[1]
        assert isinstance(call_kwargs['bootstrap_servers'], list)
        assert len(call_kwargs['bootstrap_servers']) == 2


class TestKafkaManagerClose:
    """Tests for KafkaManager close method."""
    
    @patch('src.utils.kafka_base.KAFKA_INSTALLED', True)
    def test_close_with_producer(self):
        """Test closing manager with active producer."""
        # Reset singleton for test
        KafkaManager._instance = None
        manager = KafkaManager()
        
        mock_producer = Mock()
        mock_producer.flush = Mock()
        mock_producer.close = Mock()
        manager.producer = mock_producer
        
        manager.close()
        
        mock_producer.flush.assert_called_once()
        mock_producer.close.assert_called_once()
        assert manager.producer is None
    
    @patch('src.utils.kafka_base.KAFKA_INSTALLED', True)
    def test_close_without_producer(self):
        """Test closing manager without producer."""
        # Reset singleton for test
        KafkaManager._instance = None
        manager = KafkaManager()
        manager.producer = None
        
        # Should not raise exception
        manager.close()
        
        assert manager.producer is None
    
    @patch('src.utils.kafka_base.KAFKA_INSTALLED', True)
    def test_close_with_flush_error(self):
        """Test closing manager when flush raises error."""
        KafkaManager._instance = None
        manager = KafkaManager()

        mock_producer = Mock()
        mock_producer.flush.side_effect = Exception("Flush error")
        mock_producer.close = Mock()
        manager.producer = mock_producer

        # Should handle error gracefully
        manager.close()

        assert manager.producer is None


class TestKafkaManagerThreadSafety:
    """Tests for KafkaManager thread safety."""
    
    @patch.dict('os.environ', {'KAFKA_BOOTSTRAP_SERVERS': 'localhost:9092'})
    @patch('src.utils.kafka_base.KAFKA_INSTALLED', True)
    @patch('src.utils.kafka_base.KafkaProducer')
    def test_concurrent_get_producer(self, mock_producer_class):
        """Test concurrent get_producer calls."""
        import threading
        
        mock_producer = Mock()
        mock_producer_class.return_value = mock_producer
        
        # Reset singleton for test
        KafkaManager._instance = None
        manager = KafkaManager()
        
        producers = []
        
        def get_prod():
            producers.append(manager.get_producer())
        
        threads = [threading.Thread(target=get_prod) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should get the same producer
        assert all(p is producers[0] for p in producers)
        # Producer should be initialized only once
        mock_producer_class.assert_called_once()


class TestConstants:
    """Tests for module constants."""
    
    def test_custom_token_separator(self):
        """Test CUSTOM_TOKEN_SEPARATOR constant."""
        assert CUSTOM_TOKEN_SEPARATOR == "$YashUnified2025$"
    
    def test_kafka_installed_flag(self):
        """Test KAFKA_INSTALLED flag."""
        assert isinstance(KAFKA_INSTALLED, bool)
