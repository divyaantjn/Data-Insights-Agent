"""
Shared pytest fixtures and configuration for all tests.
"""
import os
os.environ['LICENSE_ENFORCE'] = 'false'

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'product': ['A', 'B', 'C', 'A', 'B'],
        'sales': [100, 200, 150, 120, 180],
        'region': ['North', 'South', 'East', 'North', 'South'],
        'date': pd.date_range('2024-01-01', periods=5),
        'quantity': [10, 20, 15, 12, 18]
    })


@pytest.fixture
def large_dataframe():
    """Create a larger DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'value': np.random.randn(1000) * 100,
        'count': np.random.randint(1, 100, 1000),
        'date': pd.date_range('2023-01-01', periods=1000),
        'text': ['sample_' + str(i) for i in range(1000)]
    })


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    client.generate_text.return_value = "Test response"
    client.generate_embeddings.return_value = [[0.1] * 768]
    return client


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    client = Mock()
    client.upload_file.return_value = "s3://bucket/key"
    client.download_file.return_value = b"file content"
    client.generate_presigned_url.return_value = "https://presigned-url.com"
    return client


@pytest.fixture
def mock_db_connection():
    """Create a mock database connection."""
    conn = Mock()
    cursor = Mock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.rowcount = 0
    conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
    conn.cursor.return_value.__exit__ = Mock(return_value=False)
    return conn


@pytest.fixture
def mock_kafka_producer():
    """Create a mock Kafka producer."""
    producer = Mock()
    producer.send.return_value = Mock()
    producer.flush.return_value = None
    return producer


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'app': {
            'name': 'Test App',
            'version': '1.0.0',
            'host': '0.0.0.0',
            'port': 8000,
            'debug': False
        },
        'llm': {
            'provider': 'gemini',
            'model_name': 'gemini/gemini-2.0-flash-exp',
            'temperature': 0.7,
            'max_tokens': 2048
        },
        'aws': {
            'access_key_id': 'test_key',
            'secret_access_key': 'test_secret',
            'region': 'us-east-1',
            's3_bucket': 'test-bucket'
        }
    }


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = Mock()
    request.url.path = "/test"
    request.method = "GET"
    request.headers = {}
    request.client.host = "127.0.0.1"
    request.state = Mock()
    return request


@pytest.fixture
def mock_jwt_payload():
    """Create a mock JWT payload."""
    return {
        'sub': 'user123',
        'email': 'test@example.com',
        'preferred_username': 'testuser',
        'iss': 'https://keycloak.example.com/realms/test',
        'azp': 'test-client',
        'exp': 9999999999
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)
    os.environ['LICENSE_ENFORCE'] = 'false'


@pytest.fixture
def mock_xray_recorder():
    """Mock AWS X-Ray recorder."""
    with patch('aws_xray_sdk.core.xray_recorder') as mock:
        segment = Mock()
        segment.put_annotation = Mock()
        segment.put_metadata = Mock()
        segment.put_http_meta = Mock()
        mock.current_segment.return_value = segment
        mock.begin_segment.return_value = segment
        mock.begin_subsegment.return_value = segment
        mock.end_segment = Mock()
        mock.end_subsegment = Mock()
        yield mock


@pytest.fixture
def mock_otel_span():
    """Mock OpenTelemetry span."""
    span = Mock()
    span.is_recording.return_value = True
    span.set_attribute = Mock()
    span.set_status = Mock()
    span.record_exception = Mock()
    return span
