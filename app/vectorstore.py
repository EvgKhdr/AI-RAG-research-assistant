"""Chroma wrapper. One persistent collection; every chunk carries its document/folder ids.

Every chunk is also written to the keyword index (keywords.py), so the two stay in sync here."""
from functools import lru_cache

import chromadb

import keywords
from config import EMBED_MODEL
from db import DATA_DIR


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
    col = client.get_or_create_collection(
        "documents", metadata={"hnsw:space": "cosine", "embedding_model": EMBED_MODEL}
    )
    stored = (col.metadata or {}).get("embedding_model")
    if stored != EMBED_MODEL:
        raise RuntimeError(
            f"Vector store was built with '{stored}' but config uses '{EMBED_MODEL}'. "
            "Re-index the documents (delete data/chroma) before changing the embedding model."
        )
    if keywords.count() != col.count():  # first run after upgrading, or the index was deleted
        data = col.get(include=["documents", "metadatas"])
        keywords.rebuild([
            (t, m["doc_id"], m["folder_id"], m["filename"], m["chunk"], m["page"])
            for t, m in zip(data["documents"], data["metadatas"])
        ])
    return col


def add_chunks(doc_id: int, folder_id: int, filename: str, chunks: list[dict], vectors: list[list[float]]):
    """chunks: [{'text': str, 'page': int | None}] aligned with vectors."""
    delete_document(doc_id)  # re-indexing replaces old chunks
    if not chunks:
        return
    _collection().add(
        ids=[f"{doc_id}:{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "doc_id": doc_id,
                "folder_id": folder_id,
                "filename": filename,
                "chunk": i,
                "page": c["page"] or 0,  # 0 = no page info
            }
            for i, c in enumerate(chunks)
        ],
    )
    keywords.add([(c["text"], doc_id, folder_id, filename, i, c["page"] or 0) for i, c in enumerate(chunks)])


def delete_document(doc_id: int) -> None:
    _collection().delete(where={"doc_id": doc_id})
    keywords.delete(doc_id=doc_id)


def delete_folder(folder_id: int) -> None:
    _collection().delete(where={"folder_id": folder_id})
    keywords.delete(folder_id=folder_id)


def keyword_query(question: str, k: int, folder_ids: list[int] | None = None) -> list[dict]:
    _collection()  # makes sure the keyword index is in sync with Chroma
    return keywords.search(question, k, folder_ids)


def fill_scores(vector: list[float], hits: list[dict]) -> None:
    """Set hit['score'] (cosine similarity to the query) on hits that came from keyword search."""
    todo = [h for h in hits if h.get("score") is None]
    if not todo:
        return
    res = _collection().get(ids=[f"{h['doc_id']}:{h['chunk']}" for h in todo], include=["embeddings"])
    by_id = dict(zip(res["ids"], res["embeddings"]))
    for h in todo:
        emb = by_id.get(f"{h['doc_id']}:{h['chunk']}")
        h["score"] = sum(float(a) * b for a, b in zip(emb, vector)) if emb is not None else 0.0


def query(
    vector: list[float], k: int, folder_ids: list[int] | None = None, doc_id: int | None = None
) -> list[dict]:
    col = _collection()
    if col.count() == 0:
        return []
    where = None
    if doc_id is not None:
        where = {"doc_id": doc_id}
    elif folder_ids:
        where = {"folder_id": folder_ids[0]} if len(folder_ids) == 1 else {"folder_id": {"$in": folder_ids}}
    res = col.query(query_embeddings=[vector], n_results=k, where=where)
    return [
        {
            "text": text,
            "doc_id": meta["doc_id"],
            "filename": meta["filename"],
            "folder_id": meta["folder_id"],
            "page": meta["page"] or None,
            "chunk": meta["chunk"],
            "score": 1 - dist,  # cosine similarity
        }
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
