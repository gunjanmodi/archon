import asyncpg
import os
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager
from chunking import Chunk

# Connection pool instance
pool: Optional[asyncpg.Pool] = None

# Database configuration from environment variables
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "embeddings_db")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))


def embedding_to_pgvector(embedding: List[float]) -> str:
    """
    Convert a Python list of floats to pgvector's string format.
    
    Args:
        embedding: List of floats (e.g., [0.1, 0.2, 0.3])
        
    Returns:
        String representation for pgvector (e.g., "[0.1,0.2,0.3]")
    """
    return f"[{','.join(str(x) for x in embedding)}]"


async def init_db():
    """Initialize the database connection pool"""
    global pool
    pool = await asyncpg.create_pool(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        host=DB_HOST,
        port=DB_PORT,
        min_size=5,
        max_size=20
    )


async def close_db():
    """Close the database connection pool"""
    global pool
    if pool:
        await pool.close()


@asynccontextmanager
async def get_connection():
    """Get a connection from the pool"""
    global pool
    if pool is None:
        raise RuntimeError("Database pool not initialized")

    connection = await pool.acquire()
    try:
        yield connection
    finally:
        await pool.release(connection)


async def insert_embedding(
        text_chunk: str,
        embedding: List[float],
        document_id: str
) -> int:
    """
    Insert a text chunk with its embedding into the database.
    
    Args:
        text_chunk: The text content to store
        embedding: List of 1536 floats representing the embedding vector
        document_id: Reference to the source document
        
    Returns:
        The row id of the inserted record
    """
    async with get_connection() as conn:
        row_id = await conn.fetchval(
            """
            INSERT INTO embeddings (text_chunk, embedding, document_id)
            VALUES ($1, $2::vector, $3) RETURNING id
            """,
            text_chunk,
            embedding_to_pgvector(embedding),
            document_id
        )
    return row_id


async def insert_embeddings_batch(
    chunks: List[Chunk],
    embeddings: List[List[float]],
    document_id: str
) -> List[int]:
    """
    Insert multiple chunks with their embeddings in a single query.
    Returns list of inserted row ids.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if not chunks:
        return []

    values = []
    params = []

    for chunk, embedding in zip(chunks, embeddings):
        param_start = len(params) + 1
        values.append(
            f"(${param_start}, ${param_start + 1}::vector, ${param_start + 2})"
        )
        params.extend([
            chunk.text,
            embedding_to_pgvector(embedding),
            document_id
        ])

    query = f"""
        INSERT INTO embeddings (text_chunk, embedding, document_id)
        VALUES {", ".join(values)}
        RETURNING id
    """

    async with get_connection() as conn:
        rows = await conn.fetch(query, *params)

    return [row["id"] for row in rows]


async def search_embeddings(
        embedding: List[float],
        top_k: int = 5
) -> List[Tuple[int, str, str, float]]:
    """
    Search for similar embeddings using cosine distance.
    
    Args:
        embedding: Query embedding vector (list of 1536 floats)
        top_k: Number of top results to return
        
    Returns:
        List of tuples: (id, text_chunk, document_id, similarity_score)
        similarity_score is 1 - distance (so higher is more similar)
    """
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT id,
                   text_chunk,
                   document_id,
                   1 - (embedding <=> $1::vector) AS similarity_score
            FROM embeddings
            ORDER BY embedding <=> $1::vector
                LIMIT $2
            """,
            embedding_to_pgvector(embedding),
            top_k
        )

    return [(row['id'], row['text_chunk'], row['document_id'], row['similarity_score']) for row in rows]
