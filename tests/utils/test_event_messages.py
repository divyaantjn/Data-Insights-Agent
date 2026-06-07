"""
Tests for event_messages.py module.
"""
import pytest
from src.utils.event_messages import EventMessages


class TestEventMessages:
    """Tests for EventMessages class."""
    
    def test_task_events_exist(self):
        """Test that task event messages exist."""
        assert hasattr(EventMessages, 'TASK_RECEIVED')
        assert isinstance(EventMessages.TASK_RECEIVED, str)
        assert len(EventMessages.TASK_RECEIVED) > 0
    
    def test_progress_events_exist(self):
        """Test that progress event messages exist."""
        progress_attrs = [
            'PROGRESS_CONFIGURING_LLM',
            'PROGRESS_DOWNLOADING_FILE',
            'PROGRESS_PROCESSING_DATA',
            'PROGRESS_ANALYZING_SCHEMA',
            'PROGRESS_CLASSIFYING_QUESTION',
            'PROGRESS_GENERATING_PLOT',
            'PROGRESS_EXECUTING_ANALYSIS',
            'PROGRESS_FORMATTING_ANSWER',
            'PROGRESS_UPLOADING'
        ]
        
        for attr in progress_attrs:
            assert hasattr(EventMessages, attr)
            value = getattr(EventMessages, attr)
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test_success_events_exist(self):
        """Test that success event messages exist."""
        success_attrs = [
            'SUCCESS_ANALYSIS_COMPLETE',
            'SUCCESS_PLOT_GENERATED',
            'SUCCESS_ANSWER_GENERATED'
        ]
        
        for attr in success_attrs:
            assert hasattr(EventMessages, attr)
            value = getattr(EventMessages, attr)
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test_error_events_exist(self):
        """Test that error event messages exist."""
        error_attrs = [
            'ERROR_INVALID_REQUEST',
            'ERROR_S3_DOWNLOAD_FAILED',
            'ERROR_FILE_PROCESSING_FAILED',
            'ERROR_ANALYSIS_FAILED',
            'ERROR_PLOT_GENERATION_FAILED',
            'ERROR_SYSTEM_ERROR'
        ]
        
        for attr in error_attrs:
            assert hasattr(EventMessages, attr)
            value = getattr(EventMessages, attr)
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test_message_content_quality(self):
        """Test that messages have meaningful content."""
        # Task messages should mention task/received
        assert any(word in EventMessages.TASK_RECEIVED.lower() 
                  for word in ['task', 'received', 'analysis'])
        
        # Progress messages should indicate action
        assert any(word in EventMessages.PROGRESS_DOWNLOADING_FILE.lower() 
                  for word in ['download', 'file', 'loading'])
        
        # Success messages should indicate completion
        assert any(word in EventMessages.SUCCESS_ANALYSIS_COMPLETE.lower() 
                  for word in ['complete', 'success', 'finished'])
        
        # Error messages should indicate failure
        assert any(word in EventMessages.ERROR_ANALYSIS_FAILED.lower() 
                  for word in ['fail', 'error', 'unable'])
    
    def test_all_messages_are_strings(self):
        """Test that all message attributes are strings."""
        for attr_name in dir(EventMessages):
            if not attr_name.startswith('_'):
                attr_value = getattr(EventMessages, attr_name)
                assert isinstance(attr_value, str), f"{attr_name} should be a string"
    
    def test_messages_not_empty(self):
        """Test that no messages are empty strings."""
        for attr_name in dir(EventMessages):
            if not attr_name.startswith('_'):
                attr_value = getattr(EventMessages, attr_name)
                assert len(attr_value) > 0, f"{attr_name} should not be empty"
    
    def test_message_categories(self):
        """Test that messages are properly categorized."""
        task_messages = [attr for attr in dir(EventMessages) if attr.startswith('TASK_')]
        progress_messages = [attr for attr in dir(EventMessages) if attr.startswith('PROGRESS_')]
        success_messages = [attr for attr in dir(EventMessages) if attr.startswith('SUCCESS_')]
        error_messages = [attr for attr in dir(EventMessages) if attr.startswith('ERROR_')]
        
        assert len(task_messages) >= 1
        assert len(progress_messages) >= 5
        assert len(success_messages) >= 3
        assert len(error_messages) >= 5
