from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional


@dataclass
class EmbeddingResult:
    embeddings: List[List[float]]
    model_name: str
    input_count: int
    input_tokens: Optional[int] = None
    provider_response_id: Optional[str] = None


@dataclass
class GenerationResult:
    content: str
    model_name: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    provider_response_id: Optional[str] = None
    finish_reason: Optional[str] = None


class EmbeddingProvider(ABC):
    model_name: str

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a batch of input texts.
        """


class GenerationProvider(ABC):
    model_name: str

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> GenerationResult:
        """
        Generate a buffered response for the provided chat-style messages.
        """

    @abstractmethod
    def stream_generate(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """
        Stream response text chunks for the provided chat-style messages.
        """
