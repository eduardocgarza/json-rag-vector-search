# json-vector-search

Vector search over a JSON file of user data, all local 

Simple setup:

- JSON input for users
- Vectorize each user using sentence-transformers
- Store embeddings in FAISS
- Search with a natural language query

## 🔧 Setup

Make sure you have Python 3.8+.

```bash
git clone https://github.com/yourusername/json-vector-search.git
cd json-vector-search
pip install -r requirements.txt
```

## 📦 Files

- `users.json`: raw data (name, bio, interests, etc.)
- `index.py`: runs once to build and save the FAISS index
- `search.py`: enter a query and get the most relevant users
- `metadata.pkl`: saved user data
- `user_index.faiss`: saved vector index

## 🚀 Usage

### 1. Edit `users.json`  
Put your user data here. Format:

```json
[
  {
    "name": "Alice",
    "bio": "Engineer interested in robotics.",
    "interests": ["AI", "robotics"]
  }
]
```

### 2. Build the index

```bash
python index.py
```

### 3. Run search

```bash
python search.py
```

Then type something like:

```
Search query: robotics
```

And you’ll get back the top-matching users.

## 💡 Notes

- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- Works offline
- You can expand this to use a UI or hook it into an LLM for RAG stuff
