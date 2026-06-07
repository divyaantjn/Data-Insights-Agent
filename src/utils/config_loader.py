import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class ConfigLoader:
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        # Load YAML configs
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_dir = os.path.join(base_dir, 'config')
        
        # Load app config
        with open(os.path.join(config_dir, 'app_config.yaml'), 'r') as f:
            app_config = yaml.safe_load(f)
        
        # Load model config
        with open(os.path.join(config_dir, 'model_config.yaml'), 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Load milvus config
        with open(os.path.join(config_dir, 'milvus_config.yaml'), 'r') as f:
            milvus_config = yaml.safe_load(f)
        
        # Merge all configs
        config = {
            **app_config,  # This includes 'app', 'search', 'cache', 'session'
            **model_config,  # This includes 'llm', 'generation_params', 'embedding'
            **milvus_config,  # This includes 'milvus', 'collection', 'search'
            'aws': {
                'region': os.getenv("AWS_REGION_LAMBDA") or os.getenv("AWS_REGION", 'us-east-1'),
                's3_bucket': os.getenv('S3_BUCKET_NAME')
            }
        }
        
        return config
    
    def get_app_config(self) -> Dict[str, Any]:
        return self.config['app']
    
    def get_llm_config(self) -> Dict[str, Any]:
        return self.config['llm']
    
    def get_aws_config(self) -> Dict[str, Any]:
        return self.config['aws']