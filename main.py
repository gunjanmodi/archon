from contextlib import asynccontextmanager
import os
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI
from fastapi import Request
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from database import init_db, close_db, insert_embedding, insert_embeddings_batch, search_embeddings
from chunking import chunk_text
from prompt import build_prompt, get_prompt_template_hash
from citations import extract_citations, find_fabricated_citations
from semantic_cache import lookup_cached_response, store_cached_response
from logging_config import get_logger, set_request_id, setup_logging
from llm_provider import EmbeddingProvider, GenerationProvider
from openai_providers import OpenAIEmbeddingProvider, OpenAIGenerationProvider

MIN_SIMILARITY_THRESHOLD = 0.4 # todo: domain level answer may need different configuration query level
PROMPT_TEMPLATE_HASH = get_prompt_template_hash()
setup_logging()
logger = get_logger(__name__)


# Request/Response models
class DocumentRequest(BaseModel):
    text: str
    document_id: str


class DocumentResponse(BaseModel):
    id: int
    message: str


class IngestDocumentRequest(BaseModel):
    text: str
    document_id: str
    chunk_size: int = 500
    chunk_overlap: int = 50


class IngestDocumentResponse(BaseModel):
    document_id: str
    chunk_count: int
    inserted_ids: List[int]
    message: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    id: int
    text_chunk: str
    document_id: str
    similarity_score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    context_chunks: List[str]


# Lifespan event handler for database pool management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage database connection pool lifecycle.
    Initializes pool on startup, closes on shutdown.
    """
    # Startup
    embedding_provider_name = os.getenv("EMBEDDING_PROVIDER", "openai")
    generation_provider_name = os.getenv("GENERATION_PROVIDER", "openai")

    if embedding_provider_name != "openai":
        raise ValueError(f"Unsupported embedding provider: {embedding_provider_name}")
    if generation_provider_name != "openai":
        raise ValueError(f"Unsupported generation provider: {generation_provider_name}")

    app.state.embedding_provider = OpenAIEmbeddingProvider()
    app.state.generation_provider = OpenAIGenerationProvider()
    await init_db()
    logger.info(
        "application dependencies initialized",
        extra={
            "event": "application_dependencies_initialized",
            "embedding_provider": embedding_provider_name,
            "generation_provider": generation_provider_name,
        }
    )
    logger.info("database pool initialized", extra={"event": "database_pool_initialized"})
    
    yield
    
    # Shutdown
    await close_db()
    logger.info("database pool closed", extra={"event": "database_pool_closed"})


app = FastAPI(lifespan=lifespan)


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.embedding_provider


def get_generation_provider(request: Request) -> GenerationProvider:
    return request.app.state.generation_provider


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid4())
    set_request_id(request_id)
    start_time = perf_counter()
    logger.info(
        "request started",
        extra={
            "event": "request_started",
            "method": request.method,
            "path": request.url.path,
        }
    )
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = round((perf_counter() - start_time) * 1000, 2)
        logger.info(
            "request completed",
            extra={
                "event": "request_completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": None if response is None else response.status_code,
                "duration_ms": duration_ms,
            }
        )
        if response is not None:
            response.headers["X-Request-ID"] = request_id
        set_request_id(None)


@app.post("/documents", response_model=DocumentResponse)
async def create_document(
    request: DocumentRequest,
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider)
):
    """
    Create a new document chunk with embedding.
    Takes text and document_id, generates embedding, and stores it.
    """
    # Generate embedding from text
    embedding = embedding_provider.embed_texts([request.text]).embeddings[0]

    # Store in database
    row_id = await insert_embedding(
        text_chunk=request.text,
        embedding=embedding,
        document_id=request.document_id
    )
    
    return DocumentResponse(
        id=row_id,
        message="Document stored successfully"
    )


@app.post("/documents/ingest", response_model=IngestDocumentResponse)
async def ingest_document(
    request: IngestDocumentRequest,
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider)
):
    """
    Ingest a full document end to end by chunking, embedding, and storing it.
    """
    chunks = chunk_text(
        text=request.text,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    logger.info(
        "document chunked",
        extra={
            "event": "document_chunked",
            "document_id": request.document_id,
            "chunk_count": len(chunks),
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
        }
    )
    embeddings = embedding_provider.embed_texts([chunk.text for chunk in chunks]).embeddings
    inserted_ids = await insert_embeddings_batch(
        chunks=chunks,
        embeddings=embeddings,
        document_id=request.document_id
    )

    return IngestDocumentResponse(
        document_id=request.document_id,
        chunk_count=len(chunks),
        inserted_ids=inserted_ids,
        message="Document ingested successfully"
    )


@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider)
):
    """
    Search for similar document chunks based on query string.
    Uses cosine similarity of embeddings.
    """
    # todo: Chunker works on raw text. But it has no awareness of document structure. It'll happily cut a sentence in half, or split a heading from its paragraph. The overlap helps, but it doesn't solve it completely.
    # Generate embedding from query
    query_embedding = embedding_provider.embed_texts([request.query]).embeddings[0]
    
    # Query database for similar embeddings
    search_results = await search_embeddings(
        embedding=query_embedding,
        top_k=request.top_k
    )
    
    # Convert results to SearchResult objects
    results = [
        SearchResult(
            id=result[0],
            text_chunk=result[1],
            document_id=result[2],
            similarity_score=result[3]
        )
        for result in search_results
    ]
    
    return SearchResponse(results=results)


@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    generation_provider: GenerationProvider = Depends(get_generation_provider)
):
    """
    Run the full RAG flow: embed query, retrieve context, build prompt, and generate an answer.
    """
    logger.info(
        "ask request received",
        extra={
            "event": "ask_request_received",
            "query_preview": request.query[:100] + ("..." if len(request.query) > 100 else ""),
            "top_k": request.top_k,
        }
    )
    query_embedding_result = embedding_provider.embed_texts([request.query])
    query_embedding = query_embedding_result.embeddings[0]
    cache_hit = await lookup_cached_response(
        query_embedding=query_embedding,
        model_name=generation_provider.model_name,
        prompt_template_hash=PROMPT_TEMPLATE_HASH
    )
    if cache_hit is not None:
        logger.info(
            "ask cache hit",
            extra={
                "event": "ask_cache_hit",
                "similarity_score": cache_hit.similarity_score,
            }
        )
        return AskResponse(
            answer=cache_hit.response_text,
            context_chunks=cache_hit.response_metadata.get("context_chunks", [])
        )
    logger.info("ask cache miss", extra={"event": "ask_cache_miss"})

    search_results = await search_embeddings(
        embedding=query_embedding,
        top_k=request.top_k
    )
    context_chunks = [result[1] for result in search_results]
    top_similarity_score = search_results[0][3] if search_results else 0.0
    logger.info(
        "retrieval completed",
        extra={
            "event": "retrieval_completed",
            "retrieved_chunk_count": len(context_chunks),
            "top_similarity_score": top_similarity_score,
            "retrieval_scores": [result[3] for result in search_results],
        }
    )
    if top_similarity_score < MIN_SIMILARITY_THRESHOLD:
        logger.info(
            "ask rejected for low similarity",
            extra={
                "event": "ask_rejected_low_similarity",
                "top_similarity_score": top_similarity_score,
                "similarity_threshold": MIN_SIMILARITY_THRESHOLD,
            }
        )
        return AskResponse(
            answer="The provided context does not contain enough information to answer this reliably.",
            context_chunks=context_chunks
        )

    messages = build_prompt(
        query=request.query,
        retrieved_chunks=context_chunks
    )
    logger.info(
        "llm generation started",
        extra={
            "event": "llm_generation_started",
            "model_name": generation_provider.model_name,
            "context_chunk_count": len(context_chunks),
        }
    )
    generation_result = generation_provider.generate(messages)
    answer = generation_result.content
    cited_chunks = sorted(extract_citations(answer))
    logger.info(
        "generation completed",
        extra={
            "event": "generation_completed",
            "response_length": len(answer),
            "citation_count": len(cited_chunks),
            "input_tokens": generation_result.input_tokens,
            "output_tokens": generation_result.output_tokens,
        }
    )
    fabricated_citations = find_fabricated_citations(answer, len(context_chunks))
    if fabricated_citations:
        logger.info(
            "ask rejected for fabricated citations",
            extra={
                "event": "ask_rejected_fabricated_citations",
                "fabricated_citations": sorted(fabricated_citations),
                "retrieved_chunk_count": len(context_chunks),
            }
        )
        return AskResponse(
            answer="The system could not produce a reliably grounded answer from the retrieved context.",
            context_chunks=context_chunks
        )

    await store_cached_response(
        query_text=request.query,
        query_embedding=query_embedding,
        response_text=answer,
        response_metadata={
            "context_chunks": context_chunks,
            "top_similarity_score": top_similarity_score,
            "cited_chunks": cited_chunks,
        },
        model_name=generation_result.model_name,
        prompt_template_hash=PROMPT_TEMPLATE_HASH
    )

    logger.info(
        "ask response returned",
        extra={
            "event": "ask_response_returned",
            "response_length": len(answer),
            "citation_count": len(cited_chunks),
        }
    )
    return AskResponse(
        answer=answer,
        context_chunks=context_chunks
    )
