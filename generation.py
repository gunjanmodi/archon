import os
from typing import Dict, Iterator, List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

# Generation model configuration
GENERATION_MODEL = "gpt-4o-mini"


def generate_response(
    messages: List[Dict[str, str]],
    model: str = GENERATION_MODEL
) -> Iterator[str]:
    """
    Stream a response from an LLM using the assembled prompt messages.

    Args:
        messages: Chat-style messages containing system and user content
        model: Generation model to use

    Returns:
        An iterator of generated text chunks
    """
    stream = client.responses.create(
        model=model,
        input=messages,
        stream=True
    )
    for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta
