import json
import logging
from typing import Union, List, Dict
from .base_textgen import TextGenerator
from ...datamodel import Message, TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages
import requests
from dataclasses import asdict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Message):
            return {
                "role": obj.role,
                "content": obj.content
            }
        #elif isinstance(obj, dict):
        #    return {k: self.default(v) for k, v in obj.items() if not k.startswith('_')}
        elif isinstance(obj, list):
            return [self.default(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self.default({k: v for k, v in obj.__dict__.items() if not k.startswith('_')})
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)  # Convert to string if not JSON serializable

class OllamaTextGenerator(TextGenerator):
    def __init__(
        self,
        api_base: str = "http://localhost:11434",
        provider: str = "ollama",
        model: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)
        self.api_base = api_base
        self.model_name = model or "llama2"
        self.model_max_token_dict = get_models_maxtoken_dict(models)
        logger.debug(f"Initialized OllamaTextGenerator with api_base: {api_base}, model: {self.model_name}")

    def generate(
        self,
        messages: Union[List[Dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        logger.debug(f"Messages received: {messages}")
        logger.debug(f"Config received: {config}")

        use_cache = getattr(config, 'use_cache', False)
        model = getattr(config, 'model', None) or self.model_name
        prompt_tokens = num_tokens_from_messages(messages)
        max_tokens = max(
            self.model_max_token_dict.get(model, 409600) - prompt_tokens - 10, 200
        )
        logger.debug(f"Calculated max_tokens: {max_tokens}")

        ollama_config = {
            "model": model,
            "temperature": getattr(config, 'temperature', 0.7),
            "num_predict": max_tokens,
            "top_p": getattr(config, 'top_p', 1.0),
            "frequency_penalty": getattr(config, 'frequency_penalty', 0.0),
            "presence_penalty": getattr(config, 'presence_penalty', 0.0),
            "stream": False,
        }

        logger.debug(f"Ollama config: {ollama_config}")

        try:
            logger.debug("Attempting to serialize messages")
            for i, message in enumerate(messages):
                logger.debug(f"Message {i}: {message}")
                try:
                    json.dumps(message, cls=MessageEncoder)
                except TypeError as e:
                    logger.error(f"Error serializing message {i}: {e}")
                    logger.error(f"Problematic message content: {message}")
                    for key, value in message.items():
                        logger.debug(f"Key: {key}, Value type: {type(value)}")
                        try:
                            json.dumps(value)
                        except TypeError:
                            logger.error(f"Non-serializable value in key '{key}': {value}")

            ollama_config["messages"] = json.loads(json.dumps(messages, cls=MessageEncoder))
        except TypeError as e:
            logger.error(f"Error serializing messages: {e}")
            logger.error(f"Messages content: {messages}")
            # Attempt to serialize with default encoder as fallback
            try:
                ollama_config["messages"] = json.loads(json.dumps(messages, default=str))
            except Exception as e2:
                logger.error(f"Fallback serialization also failed: {e2}")
                raise

        logger.debug(f"Ollama config: {ollama_config}")

        self.model_name = model
        cache_key_params = ollama_config.copy()
        cache_key_params['messages'] = messages

        if use_cache:
            logger.debug("Attempting to use cache")
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                logger.debug("Cache hit, returning cached response")
                return TextGenerationResponse(**response)
            logger.debug("Cache miss, proceeding with API call")

        logger.debug(f"Sending request to Ollama API: {self.api_base}/api/chat")
        ollama_response = requests.post(f"{self.api_base}/api/chat", json=ollama_config)
        ollama_response.raise_for_status()
        ollama_data = ollama_response.json()
        logger.debug(f"Ollama response: {ollama_data}")

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
        logger.debug(f"Created TextGenerationResponse: {response}")

        if use_cache:
            logger.debug("Caching response")
            cache_request(
                cache=self.cache, params=cache_key_params, values=asdict(response)
            )
        return response

    def count_tokens(self, text) -> int:
        count = num_tokens_from_messages(text)
        logger.debug(f"Token count for text: {count}")
        return count