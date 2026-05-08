-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    text_chunk TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    document_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create semantic cache table
CREATE TABLE IF NOT EXISTS semantic_cache (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding vector(1536) NOT NULL,
    response_text TEXT NOT NULL,
    response_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_name VARCHAR(255) NOT NULL,
    prompt_template_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hit_count INTEGER NOT NULL DEFAULT 0,
    ttl_expiry TIMESTAMP
);

CREATE INDEX IF NOT EXISTS semantic_cache_model_prompt_idx
    ON semantic_cache (model_name, prompt_template_hash);

CREATE INDEX IF NOT EXISTS semantic_cache_query_embedding_idx
    ON semantic_cache
    USING ivfflat (query_embedding vector_cosine_ops)
    WITH (lists = 100);

-- Create dedicated app user with limited privileges (security best practice)
DO
$$
BEGIN
    CREATE ROLE app_user WITH PASSWORD 'app_password' LOGIN;
EXCEPTION WHEN DUPLICATE_OBJECT THEN
    -- User already exists, skip creation
    NULL;
END
$$;

-- Grant permissions to app_user
GRANT USAGE ON SCHEMA public TO app_user;
GRANT ALL PRIVILEGES ON TABLE embeddings TO app_user;
GRANT ALL PRIVILEGES ON TABLE semantic_cache TO app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;
