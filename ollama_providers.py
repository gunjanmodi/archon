import os
from typing import Dict, Iterator, List, Optional

from openai import OpenAI

from llm_provider import (
    EmbeddingProvider,
    EmbeddingResult,
    GenerationProvider,
    GenerationResult,
)

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.1:8b"


def _build_ollama_client(base_url: Optional[str] = None) -> OpenAI:
    resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    # Ollama's OpenAI-compatible endpoint does not check the API key, but the
    # client requires a non-empty value to be supplied.
    return OpenAI(base_url=resolved_base_url, api_key="ollama")


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model_name: str = EMBEDDING_MODEL,
        base_url: Optional[str] = None
    ) -> None:
        self.client = client or _build_ollama_client(base_url)
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


class OllamaGenerationProvider(GenerationProvider):
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model_name: str = GENERATION_MODEL,
        base_url: Optional[str] = None
    ) -> None:
        self.client = client or _build_ollama_client(base_url)
        self.model_name = model_name

    def generate(self, messages: List[Dict[str, str]]) -> GenerationResult:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        choice = response.choices[0]
        return GenerationResult(
            content=choice.message.content,
            model_name=self.model_name,
            input_tokens=getattr(response.usage, "prompt_tokens", None),
            output_tokens=getattr(response.usage, "completion_tokens", None),
            provider_response_id=getattr(response, "id", None),
            finish_reason=choice.finish_reason
        )

    def stream_generate(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
