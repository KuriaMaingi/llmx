# This file contains the list of providers and models that are available supported by LLMX.


from llmx.utils import load_config
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    config = load_config()
    logger.debug(f"Loaded config: {config}")

    if config is None:
        raise ValueError("Configuration not loaded properly")

    providers = config["providers"]
except Exception as e:
    logger.exception(f"Error loading configuration: {e}")
    raise
