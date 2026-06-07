from litellm import completion, embedding
from typing import List, Dict, Any
import numpy as np
from src.utils.obs import LLMUsageTracker
from src.utils.reasoning_extractor import extract_and_log_reasoning

class LiteLLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config['model_name']
        self.embedding_model = config.get('embedding_model', 'text-embedding-004')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2048)
        
        # Set API keys from environment
        self._setup_api_keys()

    
    def _setup_api_keys(self):
        # LiteLLM will automatically use environment variables
        # like GEMINI_API_KEY, OPENAI_API_KEY, etc.
        pass
    
    def generate(self, prompt: str, llm_params: Dict[str, Any], token_tracker, auth_token: str, **kwargs) -> str:
        # Note: token_tracker parameter kept for API compatibility
        try:
            response = completion(
                **llm_params,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            # Track tokens before returning content
            if auth_token:
                tracker = LLMUsageTracker(auth_token=auth_token)
                tracker.track_response(response, model_name=llm_params.get('model', ""))
            # Type ignore for response.choices access due to union type
            content = response.choices[0].message.content 
            content, _ = extract_and_log_reasoning(response_text=content, auth_token=auth_token)
            return content if content is not None else ""
        except Exception as e:
            print(f"LLM generation error: {e}")
            raise
    
    def create_embeddings(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:min(i + batch_size, len(texts))]
            
            try:
                response = embedding(
                    model=f"gemini/{self.embedding_model}",
                    input=batch
                )
                
                batch_embeddings = [item['embedding'] for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                print(f"Embedded batch {i//batch_size + 1}: {len(all_embeddings)} total")
            except Exception as e:
                print(f"Embedding error at batch {i}: {e}")
                raise
        
        return np.array(all_embeddings)