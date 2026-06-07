"""
Tests for config_loader.py module.
"""
import pytest
import os
from unittest.mock import Mock, patch, mock_open
from src.utils.config_loader import ConfigLoader


class TestConfigLoader:
    """Tests for ConfigLoader class."""
    
    @patch('src.utils.config_loader.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.join')
    @patch('os.path.dirname')
    def test_initialization(self, mock_dirname, mock_join, mock_file, mock_yaml):
        """Test ConfigLoader initialization."""
        mock_yaml.return_value = {
            'app': {'name': 'Test App', 'version': '1.0.0'},
            'llm': {'provider': 'gemini'},
            'milvus': {'host': 'localhost'}
        }
        mock_dirname.return_value = '/app/src/utils'
        mock_join.return_value = '/app/config/app_config.yaml'
        
        loader = ConfigLoader()
        
        assert loader.config is not None
    
    @patch('src.utils.config_loader.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.join')
    @patch('os.path.dirname')
    def test_get_app_config(self, mock_dirname, mock_join, mock_file, mock_yaml):
        """Test getting app configuration."""
        mock_yaml.return_value = {
            'app': {'name': 'Test App', 'version': '1.0.0', 'debug': False}
        }
        mock_dirname.return_value = '/app/src/utils'
        mock_join.return_value = '/app/config/app_config.yaml'
        
        loader = ConfigLoader()
        app_config = loader.get_app_config()
        
        assert app_config['name'] == 'Test App'
        assert app_config['version'] == '1.0.0'
    
    @patch('src.utils.config_loader.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.join')
    @patch('os.path.dirname')
    def test_get_llm_config(self, mock_dirname, mock_join, mock_file, mock_yaml):
        """Test getting LLM configuration."""
        mock_yaml.return_value = {
            'llm': {
                'provider': 'gemini',
                'model_name': 'gemini-pro',
                'temperature': 0.7
            }
        }
        mock_dirname.return_value = '/app/src/utils'
        mock_join.return_value = '/app/config/model_config.yaml'
        
        loader = ConfigLoader()
        llm_config = loader.get_llm_config()
        
        assert llm_config['provider'] == 'gemini'
        assert llm_config['temperature'] == 0.7
    
    @patch('src.utils.config_loader.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.join')
    @patch('os.path.dirname')
    @patch.dict('os.environ', {}, clear=True)
    def test_get_aws_config_defaults(self, mock_dirname, mock_join, mock_file, mock_yaml):
        """Test AWS config with default values."""
        mock_yaml.return_value = {}
        mock_dirname.return_value = '/app/src/utils'
        mock_join.return_value = '/app/config/app_config.yaml'
        
        loader = ConfigLoader()
        aws_config = loader.get_aws_config()
        
        assert aws_config['region'] == 'us-east-1'  # Default region
    
    @patch('src.utils.config_loader.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.join')
    @patch('os.path.dirname')
    def test_config_merging(self, mock_dirname, mock_join, mock_file, mock_yaml):
        """Test that configs are properly merged."""
        def yaml_side_effect(*args, **kwargs):
            # Return different configs based on file being opened
            return {
                'app': {'name': 'Test'},
                'llm': {'provider': 'gemini'},
                'milvus': {'host': 'localhost'}
            }
        
        mock_yaml.side_effect = [
            {'app': {'name': 'Test'}},
            {'llm': {'provider': 'gemini'}},
            {'milvus': {'host': 'localhost'}}
        ]
        mock_dirname.return_value = '/app/src/utils'
        mock_join.return_value = '/app/config/app_config.yaml'
        
        loader = ConfigLoader()
        
        assert 'app' in loader.config
        assert 'llm' in loader.config
        assert 'aws' in loader.config
