from llmx.generators.text.textgen import llm
from llmx.datamodel import TextGenerationConfig, Message

# Initialize the Ollama text generator
ollama_gen = llm(provider="ollama")

# Prepare a simple message
messages = [Message(role="user", content="What is the height of eiffel tower?")]

# Create a configuration
config = TextGenerationConfig(
    model="llama3.1:8b",
    temperature=0.7,
    max_tokens=100
)

# Generate a response
response = ollama_gen.generate(messages, config)

# Print the response
print(response.text[0].content)