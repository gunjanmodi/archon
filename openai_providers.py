import os
from typing import Dict, Iterator, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from llm_provider import (
    EmbeddingProvider,
    EmbeddingResult,
    GenerationProvider,
    GenerationResult,
)

# Load environment variables from .env file
load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"


def _build_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model_name: str = EMBEDDING_MODEL
    ) -> None:
        self.client = client or _build_openai_client()
        self.model_name = model_name

    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return EmbeddingResult(
            embeddings=[item.embedding for item in response.data],
            model_name=self.model_name,
            input_count=len(texts),
            input_tokens=getattr(response.usage, "total_tokens", None),
            provider_response_id=getattr(response, "id", None)
        )


class OpenAIGenerationProvider(GenerationProvider):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model_name: str = GENERATION_MODEL
    ) -> None:
        self.client = client or _build_openai_client()
        self.model_name = model_name

    def generate(self, messages: List[Dict[str, str]]) -> GenerationResult:
        response = self.client.responses.create(
            model=self.model_name,
            input=messages
        )
        return GenerationResult(
            content=response.output_text,
            model_name=self.model_name,
            input_tokens=getattr(response.usage, "input_tokens", None),
            output_tokens=getattr(response.usage, "output_tokens", None),
            provider_response_id=getattr(response, "id", None),
            finish_reason=getattr(response, "status", None)
        )

    def stream_generate(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        # TODO: capture usage and finish metadata from the terminal stream event so
        # streaming callers can feed token accounting and completion state into the cost layer.
        stream = self.client.responses.create(
            model=self.model_name,
            input=messages,
            stream=True
        )
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
