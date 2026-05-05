from openai import OpenAI
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    # todo: what happens when OpenAI is down? Should we have a fallback or retry mechanism?
    """
    Generate embedding vectors for the given texts.
    
    Args:
        texts: Input texts to embed
        model: Embedding model to use (default: text-embedding-3-small)
        
    Returns:
        List of embeddings, one per input text
    """
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]
