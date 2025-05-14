from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import faiss
import json

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

with open("users.json") as f:
    users = json.load(f)


def format_user(user):
    return f"Name: {user['name']}\nBio: {user['bio']}\nInterests: {', '.join(user['interests'])}"


texts = [format_user(u) for u in users]
embeddings = model.encode(texts)

with open("metadata.pkl", "wb") as f:
    pickle.dump(users, f)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
faiss.write_index(index, "user_index.faiss")

print("Index built and saved.")
