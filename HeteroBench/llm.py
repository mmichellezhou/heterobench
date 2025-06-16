import os
import logging
from openai import OpenAI
from typing import Dict, Any, Optional, Tuple

# Default configurations for different models
DEFAULT_MODEL_CONFIGS = {
    "openai": {
        "gpt-4o-mini": {"temperature": 0.0, "max_tokens": 8196},
        "gpt-4o": {"temperature": 0.0, "max_tokens": 8196},
    },
    "google": {
        "gemini-2.5-flash-preview-05-20": {"temperature": 0.0, "max_tokens": 65536},
        "gemini-2.5-pro-preview-05-06": {"temperature": 0.0, "max_tokens": 65536},
    }
}

class LLM:
    """Unified LLM class that handles different providers."""
    
    def __init__(self, provider: str, model: str = None, api_key: str = None, 
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """
        Initialize the LLM with provider and model.
        
        Args:
            provider: LLM provider ("openai" or "google")
            model: Model name (provider-specific, optional)
            api_key: API key (optional, will try to get from environment if not provided)
            temperature: Sampling temperature (optional, uses model default if not provided)
            max_tokens: Maximum tokens to generate (optional, uses model default if not provided)
        """
        self.provider = provider.lower()
        self.model = model
        
        # Get API key from parameter or environment
        if api_key is None:
            if self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
            elif self.provider == "google":
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("Google API key not provided and GOOGLE_API_KEY environment variable not set")
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Set default models if not specified
        if self.model is None:
            if self.provider == "openai":
                self.model = "gpt-4o-mini"
            elif self.provider == "google":
                self.model = "gemini-2.5-flash-preview-05-20"
        
        # Get model configuration
        model_config = self._get_model_config()
        
        # Set temperature and max_tokens (use provided values or model defaults)
        self.temperature = temperature if temperature is not None else model_config["temperature"]
        self.max_tokens = max_tokens if max_tokens is not None else model_config["max_tokens"]
        
        # Initialize the client based on provider
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "google":
            # Google using OpenAI-compatible API
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for the current model.
        
        Returns:
            Dictionary with model configuration (temperature, max_tokens)
        """
        provider_configs = DEFAULT_MODEL_CONFIGS.get(self.provider, {})
        model_config = provider_configs.get(self.model)
        
        if model_config is None:
            # Use default config if model not found
            logging.warning(f"Warning: No default config found for {self.provider}/{self.model}, using defaults")
            return {"temperature": 0.0, "max_tokens": 8196}
        
        return model_config
    
    def generate_completion(self, system_prompt: str, user_prompt: str, 
                          temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Tuple[str, Optional[Dict]]:
        """
        Generate completion using the configured provider.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature (optional, uses instance default if not provided)
            max_tokens: Maximum tokens to generate (optional, uses instance default if not provided)
            
        Returns:
            Tuple of (Generated text, serialized response dictionary)
        """
        # Use provided parameters or instance defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=max_tok
            )
            
            # Convert response to a serializable dictionary
            response_dict = {
                "model": self.model,
                "provider": self.provider,
                "temperature": temp,
                "max_tokens": max_tok,
                "response": {
                    "id": response.id,
                    "choices": [
                        {
                            "index": choice.index,
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content,
                                "function_call": choice.message.function_call,
                                "tool_calls": choice.message.tool_calls
                            },
                            "finish_reason": choice.finish_reason
                        } for choice in response.choices
                    ],
                    "created": response.created,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            }
            
            return response.choices[0].message.content, response_dict
        except Exception as e:
            logging.error(f"Error calling {self.provider.upper()}: {e}")
            return "", None
    
    def update_config(self, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """
        Update the LLM configuration.
        
        Args:
            temperature: New temperature value
            max_tokens: New max_tokens value
        """
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
    
    def __str__(self) -> str:
        return f"LLM(provider={self.provider}, model={self.model}, temperature={self.temperature}, max_tokens={self.max_tokens})"
    
    def __repr__(self) -> str:
        return self.__str__() 