import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma_db")
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

# loaded once at startup — shared across all calls
_embedder    = None
_collection  = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        # small fast model — downloads ~80MB on first run
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def _get_collection():
    global _collection
    if _collection is None:
        client      = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name     = "user_memory",
            metadata = {"hnsw:space": "cosine"}
        )
    return _collection

def store(text: str, metadata: dict = None):
    """Store a piece of text in long-term vector memory."""
    embedding  = _get_embedder().encode(text).tolist()
    collection = _get_collection()
    collection.add(
        documents  = [text],
        embeddings = [embedding],
        metadatas  = [metadata or {}],
        ids        = [str(uuid.uuid4())]
    )

def retrieve(query: str, top_k: int = 3) -> list[str]:
    """Retrieve most relevant memories for a query."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embedding = _get_embedder().encode(query).tolist()
    results   = collection.query(
        query_embeddings = [embedding],
        n_results        = min(top_k, collection.count())
    )
    return results["documents"][0] if results["documents"] else []

def store_conversation_summary(session_id: str, summary: str):
    """Store a summary of a past conversation for future retrieval."""
    store(
        text     = summary,
        metadata = {
            "type":       "conversation_summary",
            "session_id": session_id
        }
    )

def store_user_fact(fact: str):
    """Store a learned fact about the user."""
    store(
        text     = fact,
        metadata = {"type": "user_fact"}
    )