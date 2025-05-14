# json-vector-search

Vector search over JSON user data with RAG capabilities via FastAPI
Semantic search over user profiles with:

- Local vector indexing using FAISS
- REST API endpoint for RAG-augmented responses
- OpenAI integration for contextual responses

## Components

- `users.json` - Source data for user profiles
- `index.py` - Builds and saves the FAISS index and user metadata
- `main.py` - FastAPI server implementing the `/prompt` endpoint
- `user_index.faiss` - Persisted vector index
- `metadata.pkl` - Serialized user profile data

## Setup

Requires Python 3.8+

```bash
cd json-vector-search
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
```

___

## How to use

### 1. Build the index

```bash
python index.py
```

### 2. Start the API server

```bash
uvicorn main:app --reload
```

### 3. Query the API

```bash
curl -X POST http://localhost:8000/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Who knows about computer vision?"}'
```

## Architecture

1. User sends query to `/prompt` endpoint
2. System encodes query using Sentence Transformers
3. FAISS performs vector similarity search to find relevant user profiles
4. Top profiles are injected into the OpenAI prompt
5. GPT-4 generates contextually relevant response
6. Response returns to the client

## Technical Details

- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)
- Vector similarity: L2 distance
- LLM: GPT-4
- Index type: FlatL2 (exhaustive search)

## Performance Considerations

- In-memory vector search suitable for datasets under ~100k profiles
- No rate limiting implemented
- Consider caching frequent queries
