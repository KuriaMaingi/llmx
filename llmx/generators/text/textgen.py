# Import necessary modules and classes
from ...utils import load_config
# from .openai_textgen import OpenAITextGenerator
# from .palm_textgen import PalmTextGenerator
from .ollama_textgen import OllamaTextGenerator
# from .cohere_textgen import CohereTextGenerator
# from .anthropic_textgen import AnthropicTextGenerator
import logging

# Set up logging for this module
logger = logging.getLogger("llmx")


def sanitize_provider(provider: str):
    """
    Normalize the provider name to a standard format.
    This function takes various input names for providers and returns a standardized name.
    """
    # if provider.lower() in ["openai", "default", "azureopenai", "azureoai"]:
    #     return "openai"
    # elif provider.lower() in ["palm", "google"]:
    #     return "palm"
    # elif provider.lower() == "cohere":
    #     return "cohere"
    if provider.lower() == "ollama":
        return "ollama"
    # elif provider.lower() in ["hf", "huggingface"]:
    #     return "hf"
    # elif provider.lower() in ["anthropic", "claude"]:
    #     return "anthropic"
    else:
        # If the provider is not recognized, raise an error
        raise ValueError(
            f"Invalid provider '{provider}'. Only 'ollama' is supported."
        )


def llm(provider: str = None, **kwargs):
    """
    Create and return a text generator based on the specified provider.
    This function sets up the configuration and initializes the appropriate text generator.
    """
    # Load the configuration file
    config = load_config()
    
    # If no provider is specified, use the default from the config or set to 'ollama'
    if provider is None:
        provider = config["model"].get("provider")
        kwargs = config["model"].get("parameters", {})
    if provider is None:
        logger.info("No provider specified. Defaulting to 'ollama'.")
        provider = "ollama"

    # Normalize the provider name
    provider = sanitize_provider(provider)

    # Get the list of available models for the provider from the config
    models = config.get("providers", {}).get(provider, {}).get("models", {})

    # Update kwargs with provider and models information
    kwargs["provider"] = kwargs.get("provider", provider)
    kwargs["models"] = kwargs.get("models", models)

    # Initialize and return the appropriate text generator based on the provider
    if provider.lower() == "ollama":
        return OllamaTextGenerator(**kwargs)
    # if provider.lower() == "openai":
    #     return OpenAITextGenerator(**kwargs)
    # elif provider.lower() == "palm":
    #     return PalmTextGenerator(**kwargs)
    # elif provider.lower() == "cohere":
    #     return CohereTextGenerator(**kwargs)
    # elif provider.lower() == "anthropic":
    #     return AnthropicTextGenerator(**kwargs)
    # elif provider.lower() == "hf":
    #     # Check if necessary packages are installed for HuggingFace
    #     try:
    #         import transformers
    #     except ImportError:
    #         raise ImportError(
    #             "Please install the `transformers` package to use the HFTextGenerator class. pip install llmx[transformers]"
    #         )

    #     try:
    #         import torch
    #     except ImportError:
    #         raise ImportError(
    #             "Please install the `torch` package to use the HFTextGenerator class. pip install llmx[transformers]"
    #         )

    #     from .hf_textgen import HFTextGenerator
    #     return HFTextGenerator(**kwargs)
    else:
        # If the provider is not recognized (which shouldn't happen due to sanitize_provider), raise an error
        raise ValueError(
            f"Invalid provider '{provider}'. Only 'ollama' is supported."
        )