import asyncio
import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple
from unittest.mock import patch

from citations import extract_citations

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from main import AskRequest, AskResponse, ask_question  # noqa: E402

FALLBACK_MESSAGE = "The provided context does not contain enough information to answer this reliably."
REJECTION_MESSAGE = "The system could not produce a reliably grounded answer from the retrieved context."


@dataclass
class EvalObservation:
    response: AskResponse
    llm_called: bool
    top_similarity_score: float


Assertion = Callable[[EvalObservation], Tuple[bool, str]]


@dataclass
class EvalCase:
    name: str
    query: str
    search_results: Sequence[Tuple[int, str, str, float]]
    generated_chunks: Sequence[str]
    assertions: Sequence[Assertion]
    negative_assertions: Sequence[Assertion]
    top_k: int = 5


def response_contains(text: str) -> Assertion:
    def assertion(observation: EvalObservation) -> Tuple[bool, str]:
        ok = text in observation.response.answer
        return ok, f"response should contain '{text}'"

    return assertion


def response_not_contains(text: str) -> Assertion:
    def assertion(observation: EvalObservation) -> Tuple[bool, str]:
        ok = text not in observation.response.answer
        return ok, f"response should not contain '{text}'"

    return assertion


def cites_at_least_one_chunk() -> Assertion:
    def assertion(observation: EvalObservation) -> Tuple[bool, str]:
        ok = bool(extract_citations(observation.response.answer))
        return ok, "response should cite at least one chunk"

    return assertion


def response_word_count_under(limit: int) -> Assertion:
    def assertion(observation: EvalObservation) -> Tuple[bool, str]:
        ok = len(observation.response.answer.split()) <= limit
        return ok, f"response should be under {limit} words"

    return assertion


def top_similarity_at_least(minimum: float) -> Assertion:
    def assertion(observation: EvalObservation) -> Tuple[bool, str]:
        ok = observation.top_similarity_score >= minimum
        return ok, f"top similarity should be at least {minimum}"

    return assertion


def llm_call_count_is(expected: bool) -> Assertion:
    def assertion(observation: EvalObservation) -> Tuple[bool, str]:
        ok = observation.llm_called is expected
        description = "LLM should be called" if expected else "LLM should not be called"
        return ok, description

    return assertion


def acknowledges_uncertainty() -> Assertion:
    def assertion(observation: EvalObservation) -> Tuple[bool, str]:
        lower_answer = observation.response.answer.lower()
        uncertainty_markers = [
            "does not specify",
            "does not say",
            "suggests",
            "unclear",
            "not enough information",
        ]
        ok = any(marker in lower_answer for marker in uncertainty_markers)
        return ok, "response should acknowledge uncertainty"

    return assertion


async def run_eval_case(case: EvalCase) -> Tuple[bool, List[str]]:
    llm_called = False

    def fake_get_embedding(_: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3]]

    async def fake_search_embeddings(
        embedding: List[float],
        top_k: int = 5
    ) -> Sequence[Tuple[int, str, str, float]]:
        return case.search_results[:top_k]

    def fake_generate_response(_: List[dict]) -> Iterable[str]:
        nonlocal llm_called
        llm_called = True
        return iter(case.generated_chunks)

    async def fake_lookup_cached_response(
        query_embedding: List[float],
        model_name: str,
        prompt_template_hash: str,
        similarity_threshold: float = 0.95
    ):
        return None

    async def fake_store_cached_response(
        query_text: str,
        query_embedding: List[float],
        response_text: str,
        response_metadata: dict,
        model_name: str,
        prompt_template_hash: str
    ) -> int:
        return 1

    with patch("main.get_embedding", fake_get_embedding), patch(
        "main.search_embeddings", fake_search_embeddings
    ), patch("main.generate_response", fake_generate_response), patch(
        "main.lookup_cached_response", fake_lookup_cached_response
    ), patch("main.store_cached_response", fake_store_cached_response):
        response = await ask_question(AskRequest(query=case.query, top_k=case.top_k))

    observation = EvalObservation(
        response=response,
        llm_called=llm_called,
        top_similarity_score=case.search_results[0][3] if case.search_results else 0.0,
    )

    failures = []
    for assertion in case.assertions:
        ok, message = assertion(observation)
        if not ok:
            failures.append(message)

    for assertion in case.negative_assertions:
        ok, message = assertion(observation)
        if ok:
            failures.append(f"negative assertion failed: {message}")

    return not failures, failures


def build_eval_suite() -> Sequence[EvalCase]:
    return [
        EvalCase(
            name="happy_path_grounded_answer",
            query="What is the capital of France?",
            search_results=[
                (1, "France's capital city is Paris.", "doc-1", 0.86),
                (2, "France is a country in Europe.", "doc-1", 0.74),
            ],
            generated_chunks=["Paris is the capital of France [Chunk 1]."],
            assertions=[
                response_contains("Paris"),
                cites_at_least_one_chunk(),
                response_word_count_under(30),
                top_similarity_at_least(0.5),
            ],
            negative_assertions=[
                response_contains(FALLBACK_MESSAGE),
                response_contains(REJECTION_MESSAGE),
            ],
        ),
        EvalCase(
            name="out_of_scope_query_triggers_fallback",
            query="How do I rotate AWS IAM access keys?",
            search_results=[
                (1, "France's capital city is Paris.", "doc-1", 0.18),
            ],
            generated_chunks=["This should never be generated."],
            assertions=[
                response_contains(FALLBACK_MESSAGE),
                llm_call_count_is(False),
            ],
            negative_assertions=[
                cites_at_least_one_chunk(),
            ],
        ),
        EvalCase(
            name="ambiguous_query_gets_qualified_answer",
            query="What is the retry policy?",
            search_results=[
                (1, "The service may retry requests, but the exact retry count is not specified.", "doc-2", 0.58),
                (2, "The service sends embedding requests to OpenAI.", "doc-2", 0.44),
            ],
            generated_chunks=[
                "The context suggests there may be retries, but it does not specify the exact retry policy [Chunk 1]."
            ],
            assertions=[
                acknowledges_uncertainty(),
                cites_at_least_one_chunk(),
                top_similarity_at_least(0.4),
            ],
            negative_assertions=[
                response_contains(FALLBACK_MESSAGE),
                response_contains("3 retries"),
            ],
        ),
        EvalCase(
            name="fabricated_citation_triggers_rejection",
            query="What is the capital of France?",
            search_results=[
                (1, "France's capital city is Paris.", "doc-1", 0.84),
                (2, "France is a country in Europe.", "doc-1", 0.71),
            ],
            generated_chunks=["Paris is the capital of France [Chunk 5]."],
            assertions=[
                response_contains(REJECTION_MESSAGE),
                llm_call_count_is(True),
                top_similarity_at_least(0.4),
            ],
            negative_assertions=[
                response_contains(FALLBACK_MESSAGE),
                response_contains("Paris is the capital of France [Chunk 5]."),
            ],
        ),
    ]


async def run_eval_suite(cases: Sequence[EvalCase]) -> int:
    failures = 0

    for case in cases:
        passed, failure_messages = await run_eval_case(case)
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {case.name}")
        for failure_message in failure_messages:
            print(f"  - {failure_message}")
        if not passed:
            failures += 1

    print(f"\n{len(cases) - failures}/{len(cases)} evals passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_eval_suite(build_eval_suite())))
