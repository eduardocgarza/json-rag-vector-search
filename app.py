from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os
from openai import OpenAI

# Setup OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load model + index
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
index = faiss.read_index("user_index.faiss")

# Load metadata
with open("metadata.pkl", "rb") as f:
    users = pickle.load(f)

# FastAPI
app = FastAPI()


class PromptRequest(BaseModel):
    prompt: str


def format_user(user):
    return f"Name: {user['name']}\nBio: {user['bio']}\nInterests: {', '.join(user['interests'])}"


@app.post("/prompt")
async def rag_prompt(request: PromptRequest):
    query = request.prompt
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), k=3)
    top_users = [users[i] for i in I[0]]
    user_context = "\n\n".join([format_user(u) for u in top_users])

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Here are some user profiles:\n\n{user_context}",
                },
                {"role": "user", "content": query},
            ],
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
