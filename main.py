from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from embedding import get_embedding
from database import init_db, close_db, insert_embedding, insert_embeddings_batch, search_embeddings
from chunking import chunk_text
from prompt import build_prompt, get_prompt_template_hash
from generation import GENERATION_MODEL, generate_response
from citations import extract_citations, find_fabricated_citations
from semantic_cache import lookup_cached_response, store_cached_response

MIN_SIMILARITY_THRESHOLD = 0.4 # todo: domain level answer may need different configuration query level
PROMPT_TEMPLATE_HASH = get_prompt_template_hash()


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
    await init_db()
    print("Database pool initialized")
    
    yield
    
    # Shutdown
    await close_db()
    print("Database pool closed")


app = FastAPI(lifespan=lifespan)


@app.post("/documents", response_model=DocumentResponse)
async def create_document(request: DocumentRequest):
    """
    Create a new document chunk with embedding.
    Takes text and document_id, generates embedding, and stores it.
    """
    # Generate embedding from text
    embedding = get_embedding([request.text])[0]

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
async def ingest_document(request: IngestDocumentRequest):
    """
    Ingest a full document end to end by chunking, embedding, and storing it.
    """
    chunks = chunk_text(
        text=request.text,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    print("Chunks: ", chunks)
    embeddings = get_embedding([chunk.text for chunk in chunks])
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
async def search_documents(request: SearchRequest):
    """
    Search for similar document chunks based on query string.
    Uses cosine similarity of embeddings.
    """
    # todo: Chunker works on raw text. But it has no awareness of document structure. It'll happily cut a sentence in half, or split a heading from its paragraph. The overlap helps, but it doesn't solve it completely.
    # Generate embedding from query
    query_embedding = get_embedding([request.query])[0]
    
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
async def ask_question(request: AskRequest):
    """
    Run the full RAG flow: embed query, retrieve context, build prompt, and generate an answer.
    """
    query_embedding = get_embedding([request.query])[0]
    cache_hit = await lookup_cached_response(
        query_embedding=query_embedding,
        model_name=GENERATION_MODEL,
        prompt_template_hash=PROMPT_TEMPLATE_HASH
    )
    if cache_hit is not None:
        return AskResponse(
            answer=cache_hit.response_text,
            context_chunks=cache_hit.response_metadata.get("context_chunks", [])
        )

    search_results = await search_embeddings(
        embedding=query_embedding,
        top_k=request.top_k
    )
    context_chunks = [result[1] for result in search_results]
    top_similarity_score = search_results[0][3] if search_results else 0.0
    if top_similarity_score < MIN_SIMILARITY_THRESHOLD:
        return AskResponse(
            answer="The provided context does not contain enough information to answer this reliably.",
            context_chunks=context_chunks
        )

    messages = build_prompt(
        query=request.query,
        retrieved_chunks=context_chunks
    )
    answer = "".join(generate_response(messages))
    fabricated_citations = find_fabricated_citations(answer, len(context_chunks))
    if fabricated_citations:
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
            "cited_chunks": sorted(extract_citations(answer)),
        },
        model_name=GENERATION_MODEL,
        prompt_template_hash=PROMPT_TEMPLATE_HASH
    )

    return AskResponse(
        answer=answer,
        context_chunks=context_chunks
    )
