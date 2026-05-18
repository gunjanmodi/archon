import asyncio
import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import main


@pytest.fixture
def ask_request_factory():
    def _make(query: str = "What is the capital of France?", top_k: int = 5) -> main.AskRequest:
        return main.AskRequest(query=query, top_k=top_k)

    return _make


@pytest.fixture
def base_search_results():
    return [
        (1, "France's capital city is Paris.", "doc-1", 0.86),
        (2, "France is a country in Europe.", "doc-1", 0.74),
    ]


@pytest.fixture
def stubbed_ask_dependencies(monkeypatch):
    call_state = {
        "llm_called": False,
        "store_called": False,
        "stored_payload": None,
    }

    async def fake_lookup_cached_response(
        query_embedding,
        model_name,
        prompt_template_hash,
        similarity_threshold=0.95
    ):
        return None

    async def fake_search_embeddings(embedding, top_k=5):
        return []

    async def fake_store_cached_response(
        query_text,
        query_embedding,
        response_text,
        response_metadata,
        model_name,
        prompt_template_hash
    ):
        call_state["store_called"] = True
        call_state["stored_payload"] = {
            "query_text": query_text,
            "response_text": response_text,
            "response_metadata": response_metadata,
            "model_name": model_name,
            "prompt_template_hash": prompt_template_hash,
        }
        return 1

    class StubEmbeddingProvider:
        model_name = "test-embedding-model"

        def embed_texts(self, texts):
            return type(
                "EmbeddingResult",
                (),
                {
                    "embeddings": [[0.1, 0.2, 0.3] for _ in texts],
                    "input_tokens": None,
                },
            )()

    class StubGenerationProvider:
        model_name = "test-generation-model"

        def generate(self, messages):
            call_state["llm_called"] = True
            return type(
                "GenerationResult",
                (),
                {
                    "content": "Paris is the capital of France [Chunk 1].",
                    "input_tokens": None,
                    "output_tokens": None,
                    "model_name": self.model_name,
                },
            )()

        def stream_generate(self, messages):
            return iter(["Paris is the capital of France [Chunk 1]."])

    monkeypatch.setattr(main, "lookup_cached_response", fake_lookup_cached_response)
    monkeypatch.setattr(main, "search_embeddings", fake_search_embeddings)
    monkeypatch.setattr(main, "store_cached_response", fake_store_cached_response)

    return call_state, StubEmbeddingProvider(), StubGenerationProvider()


def test_ask_returns_cached_response_when_semantic_cache_hits(
    monkeypatch,
    ask_request_factory,
    stubbed_ask_dependencies
):
    call_state, embedding_provider, generation_provider = stubbed_ask_dependencies

    async def fake_lookup_cached_response(
        query_embedding,
        model_name,
        prompt_template_hash,
        similarity_threshold=0.95
    ):
        return type(
            "CacheHit",
            (),
            {
                "response_text": "Cached answer",
                "response_metadata": {"context_chunks": ["cached context"]},
                "similarity_score": 0.98,
            },
        )()

    monkeypatch.setattr(main, "lookup_cached_response", fake_lookup_cached_response)

    response = asyncio.run(main.ask_question(ask_request_factory(), embedding_provider, generation_provider))

    assert response.answer == "Cached answer"
    assert response.context_chunks == ["cached context"]
    assert call_state["llm_called"] is False


def test_ask_returns_fallback_without_llm_call_when_top_similarity_is_below_threshold(
    monkeypatch,
    ask_request_factory,
    stubbed_ask_dependencies
):
    call_state, embedding_provider, generation_provider = stubbed_ask_dependencies

    async def fake_search_embeddings(embedding, top_k=5):
        return [(1, "Unrelated chunk", "doc-1", 0.12)]

    monkeypatch.setattr(main, "search_embeddings", fake_search_embeddings)

    response = asyncio.run(
        main.ask_question(
            ask_request_factory(query="How do I rotate AWS IAM keys?"),
            embedding_provider,
            generation_provider,
        )
    )

    assert response.answer == "The provided context does not contain enough information to answer this reliably."
    assert call_state["llm_called"] is False


def test_ask_rejects_answer_when_generated_citation_is_out_of_range(
    monkeypatch,
    ask_request_factory,
    base_search_results,
    stubbed_ask_dependencies
):
    call_state, embedding_provider, _ = stubbed_ask_dependencies

    async def fake_search_embeddings(embedding, top_k=5):
        return base_search_results

    class FabricatedCitationGenerationProvider:
        model_name = "test-generation-model"

        def generate(self, messages):
            call_state["llm_called"] = True
            return type(
                "GenerationResult",
                (),
                {
                    "content": "Paris is the capital of France [Chunk 5].",
                    "input_tokens": None,
                    "output_tokens": None,
                    "model_name": self.model_name,
                },
            )()

        def stream_generate(self, messages):
            return iter(["Paris is the capital of France [Chunk 5]."])

    monkeypatch.setattr(main, "search_embeddings", fake_search_embeddings)

    response = asyncio.run(
        main.ask_question(
            ask_request_factory(),
            embedding_provider,
            FabricatedCitationGenerationProvider(),
        )
    )

    assert response.answer == "The system could not produce a reliably grounded answer from the retrieved context."
    assert call_state["llm_called"] is True
    assert call_state["store_called"] is False


def test_ask_stores_successful_grounded_answer_in_semantic_cache(
    monkeypatch,
    ask_request_factory,
    base_search_results,
    stubbed_ask_dependencies
):
    call_state, embedding_provider, generation_provider = stubbed_ask_dependencies

    async def fake_search_embeddings(embedding, top_k=5):
        return base_search_results

    monkeypatch.setattr(main, "search_embeddings", fake_search_embeddings)

    response = asyncio.run(main.ask_question(ask_request_factory(), embedding_provider, generation_provider))

    assert response.answer == "Paris is the capital of France [Chunk 1]."
    assert call_state["store_called"] is True
    assert call_state["stored_payload"]["response_metadata"]["cited_chunks"] == [1]
