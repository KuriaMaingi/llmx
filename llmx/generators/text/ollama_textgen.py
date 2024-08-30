# Import necessary modules and types
from typing import Union, List, Dict
from .base_textgen import TextGenerator
from ...datamodel import Message, TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages
import os
import requests
from dataclasses import asdict
import json

# Define a custom JSON encoder for the Message class
class MessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Message):
            return {
                "role": obj.role,
                "content": obj.content
                # Add any other relevant attributes
            }
        return super().default(obj)

# Define a class for generating text using Ollama
class OllamaTextGenerator(TextGenerator):
    def __init__(
        self,
        api_base: str = "http://localhost:11434",  # Default URL for Ollama API
        provider: str = "ollama",
        model: str = None,
        models: Dict = None,
    ):
        # Initialize the parent class
        super().__init__(provider=provider)
        # Set the API base URL
        self.api_base = api_base
        # Set the default model name
        self.model_name = model or "llama2"
        # Get a dictionary of maximum token limits for different models
        self.model_max_token_dict = get_models_maxtoken_dict(models)

    # Method to generate text based on input messages
    def generate(
        self,
        messages: Union[List[dict], str],  # Input messages or text
        config: TextGenerationConfig = TextGenerationConfig(),  # Configuration for text generation
        **kwargs,
    ) -> TextGenerationResponse:
        # Determine whether to use caching
        use_cache = config.use_cache
        # Get the model name from config or use the default
        model = config.model or self.model_name
        # Count the number of tokens in the input messages
        prompt_tokens = num_tokens_from_messages(messages)
        # Calculate the maximum number of tokens for the response
        max_tokens = max(
            self.model_max_token_dict.get(
                model, 4096) - prompt_tokens - 10, 200
        )

        # Prepare the configuration for Ollama API
        ollama_config = {
            "model": model,
            "temperature": config.temperature,
            "num_predict": max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "stream": False,
            "messages": [MessageEncoder().default(msg) for msg in messages],
        }

        # Update the model name
        self.model_name = model
        # Prepare parameters for caching
        cache_key_params = ollama_config | {"messages": messages}
        
        # If caching is enabled, try to retrieve a cached response
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        # Make a POST request to the Ollama API
        ollama_response = requests.post(f"{self.api_base}/api/chat", json=ollama_config)
        # Raise an exception if the request was unsuccessful
        ollama_response.raise_for_status()
        # Parse the JSON response
        ollama_data = ollama_response.json()

        # Create a TextGenerationResponse object with the API response
        response = TextGenerationResponse(
            text=[Message(role="assistant", content=ollama_data['message']['content'])],
            logprobs=[],
            config=ollama_config,
            usage={
                "prompt_tokens": ollama_data['prompt_eval_count'],
                "completion_tokens": ollama_data['eval_count'],
                "total_tokens": ollama_data['prompt_eval_count'] + ollama_data['eval_count']
            },
        )

        # If caching is enabled, store the response in the cache
        if use_cache:
            cache_request(
                cache=self.cache, params=cache_key_params, values=asdict(response)
            )
        return response

    # Method to count the number of tokens in a given text
    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
