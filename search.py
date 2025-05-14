# search.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

# Load
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("user_index.faiss")

with open("metadata.pkl", "rb") as f:
    users = pickle.load(f)

# User query
query = input("Search query: ")
query_vector = model.encode([query])

# Search
D, I = index.search(np.array(query_vector), k=3)

print("\n🔍 Top Matches:")
for idx in I[0]:
    user = users[idx]
    print(
        f"- {user['name']}: {user['bio']} (Interests: {', '.join(user['interests'])})"
    )
