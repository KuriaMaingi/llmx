import json
import logging
from typing import Union, List, Dict
from .base_textgen import TextGenerator
from ...datamodel import Message, TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages
import requests
from dataclasses import asdict
import re

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

    def generate(
        self,
        messages: Union[List[Dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        logger.debug(f"Messages received: {messages[:100]}...")  # Truncate long messages
        logger.debug(f"Config received: {config}")

        model = config.model or self.model_name
        if not model:
            raise ValueError("No model specified")

        prompt_tokens = num_tokens_from_messages(messages)
        max_tokens = max(
            self.model_max_token_dict.get(model, 128000) - prompt_tokens - 10, 200
        )

        ollama_config = {
            "model": model,
            "temperature": getattr(config, 'temperature', 0.7),
            "num_predict": max_tokens,
            "top_p": getattr(config, 'top_p', 1.0),
            "frequency_penalty": getattr(config, 'frequency_penalty', 0.0),
            "presence_penalty": getattr(config, 'presence_penalty', 0.0),
            "stream": False,
        }

        ollama_config["messages"] = messages if isinstance(messages, list) else [{"role": "user", "content": messages}]

        logger.debug(f"Ollama config: {ollama_config}")

        self.model_name = model
        cache_key_params = ollama_config.copy()
        cache_key_params['messages'] = messages

        # Check if caching is available and enabled
        if hasattr(self, 'cache') and hasattr(config, 'use_cache') and config.use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                logger.debug("Using cached response")
                return TextGenerationResponse(**response)

        ollama_response = requests.post(f"{self.api_base}/api/chat", json=ollama_config)
        ollama_response.raise_for_status()
        ollama_data = ollama_response.json()

        logger.debug(f"Ollama response: {ollama_data}")

        content = ollama_data['message']['content']
        
        # Try to extract JSON from code block if present
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            logger.debug("Found JSON in code block")
            content = json_match.group(1)
        else:
            # If no code block, try to find a JSON array in the content
            json_match = re.search(r'\[([\s\S]*?)\]', content)
            if json_match:
                logger.debug("Found JSON array in content")
                content = f"[{json_match.group(1)}]"
            else:
                logger.debug("No JSON structure found in content")

        try:
            # Attempt to parse the content as JSON
            parsed_content = json.loads(content)
            # If successful, convert back to a formatted JSON string
            content = json.dumps(parsed_content, indent=2)
            logger.debug("Successfully parsed and formatted JSON content")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse model output as JSON: {content}")
            logger.warning("Returning original content without JSON parsing")

        response = TextGenerationResponse(
            text=[Message(role="assistant", content=content)],
            logprobs=[],
            config=ollama_config,
            usage={
                "prompt_tokens": ollama_data['prompt_eval_count'],
                "completion_tokens": ollama_data['eval_count'],
                "total_tokens": ollama_data['prompt_eval_count'] + ollama_data['eval_count']
            },
        )

        # Cache the response if caching is available and enabled
        if hasattr(self, 'cache') and hasattr(config, 'use_cache') and config.use_cache:
            logger.debug("Caching response")
            cache_request(
                cache=self.cache, params=cache_key_params, values=asdict(response)
            )

        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)