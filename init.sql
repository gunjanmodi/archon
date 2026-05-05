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
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;
