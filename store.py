import uuid
import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer
from config import *

_models = {}
def _get(name):
    if name not in _models:
        _models[name] = SentenceTransformer(name, device=EMBED_DEVICE)
    return _models[name]

_db = chromadb.PersistentClient(path=DB_DIR)
texts = _db.get_or_create_collection("texts", metadata={"hnsw:space": "cosine"})
images = _db.get_or_create_collection("images", metadata={"hnsw:space": "cosine"})

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def embed_docs(strs):
    return _get(TEXT_EMBED).encode(strs, normalize_embeddings=True, batch_size=32).tolist()

def embed_query(q):
    return embed_docs([QUERY_PREFIX + q])

def clip_images(paths):
    ims = [Image.open(p).convert("RGB") for p in paths]
    return _get(IMAGE_EMBED).encode(ims, normalize_embeddings=True).tolist()

def add_chunks(items):
    """items: list of (text, metadata)"""
    if not items:
        return
    docs = [t for t, _ in items]
    texts.add(ids=[uuid.uuid4().hex for _ in items], documents=docs,
              embeddings=embed_docs(docs), metadatas=[m for _, m in items])

def add_image(path, meta):
    try:
        emb = clip_images([path])
    except Exception:
        return False
    images.add(ids=[uuid.uuid4().hex], embeddings=emb, metadatas=[{**meta, "path": str(path)}])
    return True

def delete_source(name):
    for c in (texts, images):
        c.delete(where={"source": name})

def reset():
    for c in (texts, images):
        ids = c.get()["ids"]
        if ids:
            c.delete(ids=ids)
