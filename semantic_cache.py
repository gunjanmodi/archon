import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from database import embedding_to_pgvector, get_connection

SEMANTIC_CACHE_THRESHOLD = 0.95


@dataclass
class SemanticCacheHit:
    response_text: str
    response_metadata: Dict[str, Any]
    similarity_score: float


async def lookup_cached_response(
    query_embedding: List[float],
    model_name: str,
    prompt_template_hash: str,
    similarity_threshold: float = SEMANTIC_CACHE_THRESHOLD
) -> Optional[SemanticCacheHit]:
    """
    Look up a semantically similar cached response for the same model and prompt version.
    """
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT id,
                   response_text,
                   response_metadata,
                   1 - (query_embedding <=> $1::vector) AS similarity_score
            FROM semantic_cache
            WHERE model_name = $2
              AND prompt_template_hash = $3
              AND (ttl_expiry IS NULL OR ttl_expiry > CURRENT_TIMESTAMP)
            ORDER BY query_embedding <=> $1::vector
            LIMIT 1
            """,
            embedding_to_pgvector(query_embedding),
            model_name,
            prompt_template_hash
        )

        if row is None or row["similarity_score"] < similarity_threshold:
            return None

        await conn.execute(
            """
            UPDATE semantic_cache
            SET last_accessed_at = CURRENT_TIMESTAMP,
                hit_count = hit_count + 1
            WHERE id = $1
            """,
            row["id"]
        )

    return SemanticCacheHit(
        response_text=row["response_text"],
        response_metadata=row["response_metadata"],
        similarity_score=row["similarity_score"]
    )


async def store_cached_response(
    query_text: str,
    query_embedding: List[float],
    response_text: str,
    response_metadata: Dict[str, Any],
    model_name: str,
    prompt_template_hash: str
) -> int:
    """
    Store a generated response in the semantic cache.
    """
    async with get_connection() as conn:
        cache_id = await conn.fetchval(
            """
            INSERT INTO semantic_cache (
                query_text,
                query_embedding,
                response_text,
                response_metadata,
                model_name,
                prompt_template_hash
            )
            VALUES ($1, $2::vector, $3, $4::jsonb, $5, $6)
            RETURNING id
            """,
            query_text,
            embedding_to_pgvector(query_embedding),
            response_text,
            json.dumps(response_metadata),
            model_name,
            prompt_template_hash
        )

    return cache_id
