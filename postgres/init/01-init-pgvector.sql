-- Initialize pgvector extension for vector similarity search
-- This script runs automatically when the PostgreSQL container is first created

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify the extension is installed
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE agentic_rag TO postgres;

-- Display success message
DO $$
BEGIN
    RAISE NOTICE 'pgvector extension initialized successfully';
END
$$;
