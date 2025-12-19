### to run local 
docker compose up -d

### neo4j
MATCH (n1) - [r] -> (n2) RETURN r, n1, n2 LIMIT 1000

### extra mcp wit claude code
claude mcp add --transport sse crawl4ai-rag http://localhost:8052/sse

claude mcp add neon -s user -- npx -y @neondatabase/mcp-server-neo start [neon_api_key]

* start plan mode

### ingest
python -m ingestion.ingest
python -m agent.api
python cli.py
