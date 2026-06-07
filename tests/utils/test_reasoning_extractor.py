"""
Tests for reasoning_extractor.py module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.utils.reasoning_extractor import (
    extract_and_log_reasoning,
    REASONING_SECTION_PROMPT
)


class TestExtractAndLogReasoning:
    """Tests for extract_and_log_reasoning(response_text, auth_token="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9") function."""
    
    def test_extract_reasoning_with_section(self):
        """Test extracting reasoning when REASONING section exists."""
        response_text = """
This is the main response content.

REASONING:
I analyzed the data and found patterns.
The conclusion is based on statistical analysis.
"""
        
        with patch('src.utils.reasoning_extractor.create_reasoning_logger') as mock_logger_factory:
            mock_logger = Mock()
            mock_logger_factory.return_value = mock_logger
            
            cleaned, reasoning = extract_and_log_reasoning(response_text, auth_token="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
            
            assert "This is the main response content." in cleaned
            assert "REASONING:" not in cleaned
            assert "I analyzed the data" in reasoning
            assert "statistical analysis" in reasoning
            mock_logger.log_reasoning.assert_called_once()
    
    def test_extract_reasoning_case_insensitive(self):
        """Test extracting reasoning with different case."""
        response_text = """
Main content here.

reasoning:
This is my reasoning.
"""
        
        with patch('src.utils.reasoning_extractor.create_reasoning_logger') as mock_logger_factory:
            mock_logger = Mock()
            mock_logger_factory.return_value = mock_logger
            
            cleaned, reasoning = extract_and_log_reasoning(response_text, auth_token="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
            
            assert "reasoning:" not in cleaned.lower()
            assert "This is my reasoning" in reasoning
    
    def test_extract_reasoning_no_section(self):
        """Test extracting reasoning when no REASONING section exists."""
        response_text = "This is just a regular response without reasoning."
        
        with patch('src.utils.reasoning_extractor.create_reasoning_logger') as mock_logger_factory:
            mock_logger = Mock()
            mock_logger_factory.return_value = mock_logger
            
            cleaned, reasoning = extract_and_log_reasoning(response_text, auth_token="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
            
            assert cleaned == response_text
            assert reasoning == ""
            mock_logger.log_reasoning.assert_not_called()
    
    def test_extract_reasoning_with_auth_token(self):
        """Test extracting reasoning with auth token."""
        response_text = """
Response content.

REASONING:
My reasoning here.
"""
        auth_token = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        
        with patch('src.utils.reasoning_extractor.create_reasoning_logger') as mock_logger_factory:
            mock_logger = Mock()
            mock_logger_factory.return_value = mock_logger
            
            cleaned, reasoning = extract_and_log_reasoning(response_text, auth_token="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
            
            mock_logger.log_reasoning.assert_called_once_with("My reasoning here.", auth_token)
    
    def test_extract_reasoning_multiline(self):
        """Test extracting multiline reasoning."""
        response_text = """
Answer to the question.

REASONING:
Line 1 of reasoning.
Line 2 of reasoning.
Line 3 of reasoning.
"""
        
        with patch('src.utils.reasoning_extractor.create_reasoning_logger') as mock_logger_factory:
            mock_logger = Mock()
            mock_logger_factory.return_value = mock_logger
            
            cleaned, reasoning = extract_and_log_reasoning(response_text, auth_token="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
            
            assert "Line 1" in reasoning
            assert "Line 2" in reasoning
            assert "Line 3" in reasoning
    
    def test_extract_reasoning_empty_section(self):
        """Test extracting reasoning with empty REASONING section."""
        response_text = """
Response content.

REASONING:
"""
        
        with patch('src.utils.reasoning_extractor.create_reasoning_logger') as mock_logger_factory:
            mock_logger = Mock()
            mock_logger_factory.return_value = mock_logger
            
            cleaned, reasoning = extract_and_log_reasoning(response_text, auth_token="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
            
            assert reasoning == ""
            mock_logger.log_reasoning.assert_not_called()
    
    def test_extract_reasoning_logger_exception(self):
        """Test extracting reasoning when logger raises exception."""
        response_text = """
Content.

REASONING:
Some reasoning.
"""
        
        with patch('src.utils.reasoning_extractor.create_reasoning_logger') as mock_logger_factory:
            mock_logger = Mock()
            mock_logger.log_reasoning.side_effect = Exception("Kafka error")
            mock_logger_factory.return_value = mock_logger
            
            # Should not raise exception
            cleaned, reasoning = extract_and_log_reasoning(response_text, auth_token="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
            
            assert reasoning == "Some reasoning."
    
    def test_extract_reasoning_whitespace_handling(self):
        """Test extracting reasoning with various whitespace."""
        response_text = """
Content here.

   REASONING:   
   Reasoning with extra whitespace.   
"""
        
        with patch('src.utils.reasoning_extractor.create_reasoning_logger') as mock_logger_factory:
            mock_logger = Mock()
            mock_logger_factory.return_value = mock_logger
            
            cleaned, reasoning = extract_and_log_reasoning(response_text, auth_token="Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
            
            assert reasoning == "Reasoning with extra whitespace."
            assert "REASONING:" not in cleaned


class TestReasoningSectionPrompt:
    """Tests for REASONING_SECTION_PROMPT constant."""

    def test_reasoning_prompt_exists(self):
        """Test that REASONING_SECTION_PROMPT is defined and is a string."""
        assert REASONING_SECTION_PROMPT is not None
        assert isinstance(REASONING_SECTION_PROMPT, str)

    def test_reasoning_prompt_is_string_type(self):
        """Test that REASONING_SECTION_PROMPT is exactly a str instance."""
        assert type(REASONING_SECTION_PROMPT) is str

    def test_reasoning_prompt_is_importable(self):
        """Test that REASONING_SECTION_PROMPT can be imported from the module."""
        from src.utils.reasoning_extractor import REASONING_SECTION_PROMPT as rsp
        assert rsp is not None