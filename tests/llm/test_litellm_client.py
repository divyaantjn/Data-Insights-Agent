"""
Tests for litellm_client.py module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.llm.litellm_client import LiteLLMClient


class TestLiteLLMClient:
    """Tests for LiteLLMClient class."""

    @pytest.fixture
    def llm_config(self):
        """Create LLM configuration."""
        return {
            'provider': 'gemini',
            'model_name': 'gemini/gemini-2.0-flash-exp',
            'temperature': 0.7,
            'max_tokens': 2048,
            'top_p': 0.95,
            'timeout': 30,
            'max_retries': 3,
            'embedding_model': 'text-embedding-004'
        }

    @pytest.fixture
    def llm_params(self):
        """Create LLM params dict passed to generate()."""
        return {
            'model': 'gemini/gemini-2.0-flash-exp',
            'temperature': 0.7,
            'max_tokens': 2048,
        }

    @pytest.fixture
    def client(self, llm_config):
        """Create LiteLLMClient instance."""
        return LiteLLMClient(llm_config)

    @pytest.fixture
    def token_tracker(self):
        """Create a mock token tracker."""
        return Mock()

    # ------------------------------------------------------------------ #
    #  Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def test_client_initialization(self, client, llm_config):
        """Test client initialization."""
        assert client.config == llm_config
        assert client.model_name == llm_config['model_name']
        assert client.temperature == llm_config['temperature']

    def test_client_config_access(self, client, llm_config):
        """Test accessing configuration values."""
        assert client.config['provider'] == 'gemini'
        assert client.config['max_tokens'] == 2048
        assert client.config['timeout'] == 30

    # ------------------------------------------------------------------ #
    #  generate()                                                           #
    # ------------------------------------------------------------------ #

    @patch('src.llm.litellm_client.completion')
    @patch('src.llm.litellm_client.extract_and_log_reasoning', return_value=("Generated text", None))
    @patch('src.llm.litellm_client.LLMUsageTracker')
    def test_generate_basic(self, mock_tracker_cls, mock_extract, mock_completion, client, llm_params, token_tracker):
        """Test basic text generation."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated text"))]
        mock_completion.return_value = mock_response

        result = client.generate("Test prompt", llm_params, token_tracker, "auth-token-123")

        assert result == "Generated text"
        mock_completion.assert_called_once()

    @patch('src.llm.litellm_client.completion')
    @patch('src.llm.litellm_client.extract_and_log_reasoning', return_value=("Response", None))
    @patch('src.llm.litellm_client.LLMUsageTracker')
    def test_generate_with_system_prompt(self, mock_tracker_cls, mock_extract, mock_completion, client, llm_params, token_tracker):
        """Test text generation with system prompt passed via kwargs."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_completion.return_value = mock_response

        result = client.generate(
            "User prompt",
            llm_params,
            token_tracker,
            "auth-token-123",
            system_prompt="You are a helpful assistant",
        )

        assert result == "Response"

    @patch('src.llm.litellm_client.completion')
    @patch('src.llm.litellm_client.extract_and_log_reasoning', return_value=("Response", None))
    @patch('src.llm.litellm_client.LLMUsageTracker')
    def test_generate_with_custom_temperature(self, mock_tracker_cls, mock_extract, mock_completion, client, token_tracker):
        """Test text generation with custom temperature inside llm_params."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_completion.return_value = mock_response

        custom_llm_params = {
            'model': 'gemini/gemini-2.0-flash-exp',
            'temperature': 0.9,
            'max_tokens': 2048,
        }
        client.generate("Prompt", custom_llm_params, token_tracker, "auth-token-123")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs['temperature'] == 0.9

    @patch('src.llm.litellm_client.completion')
    def test_generate_error_handling(self, mock_completion, client, llm_params, token_tracker):
        """Test error handling in text generation."""
        mock_completion.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            client.generate("Prompt", llm_params, token_tracker, "auth-token-123")

    @patch('src.llm.litellm_client.completion')
    @patch('src.llm.litellm_client.extract_and_log_reasoning', return_value=("Extracted content", None))
    @patch('src.llm.litellm_client.LLMUsageTracker')
    def test_generate_response_format(self, mock_tracker_cls, mock_extract, mock_completion, client, llm_params, token_tracker):
        """Test that response is properly extracted as a string."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Extracted content"))]
        mock_completion.return_value = mock_response

        result = client.generate("Prompt", llm_params, token_tracker, "auth-token-123")

        assert isinstance(result, str)
        assert result == "Extracted content"

    @patch('src.llm.litellm_client.completion')
    @patch('src.llm.litellm_client.extract_and_log_reasoning', return_value=("", None))
    @patch('src.llm.litellm_client.LLMUsageTracker')
    def test_generate_empty_response(self, mock_tracker_cls, mock_extract, mock_completion, client, llm_params, token_tracker):
        """Test handling of empty response content."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=""))]
        mock_completion.return_value = mock_response

        result = client.generate("Prompt", llm_params, token_tracker, "auth-token-123")

        assert result == ""

    @patch('src.llm.litellm_client.completion')
    @patch('src.llm.litellm_client.extract_and_log_reasoning', return_value=("Response", None))
    @patch('src.llm.litellm_client.LLMUsageTracker')
    def test_generate_with_max_tokens(self, mock_tracker_cls, mock_extract, mock_completion, client, token_tracker):
        """Test text generation with custom max_tokens inside llm_params."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_completion.return_value = mock_response

        custom_llm_params = {
            'model': 'gemini/gemini-2.0-flash-exp',
            'temperature': 0.7,
            'max_tokens': 1000,
        }
        client.generate("Prompt", custom_llm_params, token_tracker, "auth-token-123")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs['max_tokens'] == 1000

    @patch('src.llm.litellm_client.completion')
    @patch('src.llm.litellm_client.extract_and_log_reasoning', return_value=("Tracked", None))
    def test_generate_tracks_tokens_when_auth_token_provided(self, mock_extract, mock_completion, client, llm_params, token_tracker):
        """Test that LLMUsageTracker is called when auth_token is provided."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Tracked"))]
        mock_completion.return_value = mock_response

        with patch('src.llm.litellm_client.LLMUsageTracker') as mock_tracker_cls:
            mock_tracker_instance = Mock()
            mock_tracker_cls.return_value = mock_tracker_instance

            client.generate("Prompt", llm_params, token_tracker, "auth-token-xyz")

            mock_tracker_cls.assert_called_once_with(auth_token="auth-token-xyz")
            mock_tracker_instance.track_response.assert_called_once()

    @patch('src.llm.litellm_client.completion')
    @patch('src.llm.litellm_client.extract_and_log_reasoning', return_value=("No track", None))
    def test_generate_skips_tracking_when_no_auth_token(self, mock_extract, mock_completion, client, llm_params, token_tracker):
        """Test that LLMUsageTracker is NOT called when auth_token is empty/falsy."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="No track"))]
        mock_completion.return_value = mock_response

        with patch('src.llm.litellm_client.LLMUsageTracker') as mock_tracker_cls:
            client.generate("Prompt", llm_params, token_tracker, "")

            mock_tracker_cls.assert_not_called()

    # ------------------------------------------------------------------ #
    #  create_embeddings()                                                  #
    # ------------------------------------------------------------------ #

    @patch('src.llm.litellm_client.embedding')
    def test_create_embeddings_single(self, mock_embedding, client):
        """Test generating embeddings for a single text."""
        mock_response = Mock()
        mock_response.data = [{'embedding': [0.1, 0.2, 0.3]}]
        mock_embedding.return_value = mock_response

        result = client.create_embeddings(["Test text"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])
        mock_embedding.assert_called_once()

    @patch('src.llm.litellm_client.embedding')
    def test_create_embeddings_batch(self, mock_embedding, client):
        """Test generating embeddings for a batch of texts."""
        mock_response = Mock()
        mock_response.data = [
            {'embedding': [0.1, 0.2]},
            {'embedding': [0.3, 0.4]},
        ]
        mock_embedding.return_value = mock_response

        result = client.create_embeddings(["Text 1", "Text 2"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2])
        np.testing.assert_array_almost_equal(result[1], [0.3, 0.4])

    @patch('src.llm.litellm_client.embedding')
    def test_create_embeddings_empty_input(self, mock_embedding, client):
        """Test that empty input returns an empty numpy array without calling the API."""
        result = client.create_embeddings([])

        assert isinstance(result, np.ndarray)
        assert result.size == 0
        mock_embedding.assert_not_called()

    @patch('src.llm.litellm_client.embedding')
    def test_create_embeddings_error(self, mock_embedding, client):
        """Test error handling in embedding generation."""
        mock_embedding.side_effect = Exception("Embedding API Error")

        with pytest.raises(Exception, match="Embedding API Error"):
            client.create_embeddings(["Test text"])

    @patch('src.llm.litellm_client.embedding')
    def test_create_embeddings_uses_configured_model(self, mock_embedding, client):
        """Test that create_embeddings uses the model from config."""
        mock_response = Mock()
        mock_response.data = [{'embedding': [0.1, 0.2]}]
        mock_embedding.return_value = mock_response

        client.create_embeddings(["Text"])

        call_kwargs = mock_embedding.call_args[1]
        assert 'text-embedding-004' in call_kwargs['model']

    @patch('src.llm.litellm_client.embedding')
    def test_create_embeddings_respects_batch_size(self, mock_embedding, client):
        """Test that large inputs are split into batches correctly."""
        mock_response = Mock()
        mock_response.data = [{'embedding': [0.1, 0.2]}] * 3
        mock_embedding.return_value = mock_response

        texts = [f"Text {i}" for i in range(7)]
        # batch_size=3 → ceil(7/3) = 3 API calls
        client.create_embeddings(texts, batch_size=3)

        assert mock_embedding.call_count == 3