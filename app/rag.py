"""Retrieval-augmented prompting: fetch relevant chunks and build the LLM messages."""
import re

import embeddings
import folders
import vectorstore
from config import CONTEXT_CHARS, KEYWORD_SLOTS, MAX_CHUNKS, MIN_SCORE, REL_CUTOFF, RRF_K

_RULES_STRICT = """- Answer ONLY from the sources. Do not use outside knowledge, even if you know the answer.
- If the sources do not contain the answer, say so plainly and say which documents you checked."""
_RULES_GENERAL = """- Prefer the sources. If they do not contain the answer, say so plainly; you may then add general knowledge, clearly labelled as not coming from the documents."""

SYSTEM_PROMPT = """You are a research assistant that answers questions using the user's documents.
{inventory}

Relevant excerpts are provided as numbered sources [1], [2], ... in the user's latest message.
{rules}
- Cite sources inline like [1] or [2][3]. Do not invent citations.
- The sources are only the best-matching excerpts, not whole documents. If a listed document has no excerpt on the topic, say it was not found in the retrieved excerpts rather than claiming the document does not cover it.
- Questions about which or how many documents you have are answered from the document list above.
- Be concise."""

MAX_LISTED = 100


def inventory(docs: list[dict]) -> str:
    """Document list for the system prompt, so the model knows what it can search."""
    ready = [d for d in docs if d["status"] == "indexed"]
    if not ready:
        return "The user has no searchable documents in scope."
    lines = [f"Documents in scope ({len(ready)} searchable):"]
    lines += [f"- {d['filename']} (folder: {d['folder']})" for d in ready[:MAX_LISTED]]
    if len(ready) > MAX_LISTED:
        lines.append(f"- ... and {len(ready) - MAX_LISTED} more")
    if len(ready) < len(docs):
        lines.append(f"({len(docs) - len(ready)} more document(s) are not searchable: indexing failed or pending.)")
    return "\n".join(lines)


# Wording that asks about the whole collection rather than one fact. Such questions score low against
# every chunk, so for them each document's best chunks bypass the MIN_SCORE floor.
_BROAD = re.compile(
    r"\b(all|every|each|overview|summar\w*|compar\w*|differences?|documents|files|sources)\b", re.I
)


def _key(h: dict) -> tuple[int, int]:
    return h["doc_id"], h["chunk"]


def retrieve(
    question: str,
    folder_ids: list[int] | None = None,
    budget: int = CONTEXT_CHARS,
    docs: list[dict] | None = None,
) -> list[dict]:
    """Hybrid search; the number of chunks depends on the question, not a fixed k.

    Semantic (embedding) and keyword (BM25) results are merged with reciprocal rank fusion. Chunks
    are then added until the character budget or MAX_CHUNKS is reached, in this order:
    1. the best keyword matches (exact terms, names, numbers), whatever their cosine score;
    2. the best chunk of each document not yet represented, if it is relevant enough
       (>= MIN_SCORE), so one document cannot crowd out the rest; for broad questions ("summarize
       each document") the best two chunks of every document, whatever their score;
    3. the fused ranking, but only chunks scoring >= max(MIN_SCORE, REL_CUTOFF x best score).
    A narrow question therefore gets a few chunks and a broad one many. May return nothing.
    Each hit gets 'via': 'semantic', 'keyword' or 'both'. folder_ids=None searches every folder."""
    vector = embeddings.embed([question])[0]
    if docs is None:
        docs = folders.scope_documents(folder_ids)
    doc_ids = [d["id"] for d in docs if d["status"] == "indexed"]
    pool = 2 * MAX_CHUNKS
    semantic = vectorstore.query(vector, pool, folder_ids)
    keyword = vectorstore.keyword_query(question, pool, folder_ids)

    fused: dict[tuple[int, int], float] = {}
    for results in (semantic, keyword):
        for rank, h in enumerate(results, 1):
            fused[_key(h)] = fused.get(_key(h), 0.0) + 1 / (RRF_K + rank)
    candidates = {_key(h): h for h in semantic + keyword}
    sem_keys = {_key(h) for h in semantic}
    kw_keys = {_key(h) for h in keyword}
    vectorstore.fill_scores(vector, list(candidates.values()))  # keyword hits have no cosine score yet
    best = max((h["score"] for h in candidates.values()), default=0.0)
    cutoff = max(MIN_SCORE, REL_CUTOFF * best)

    chosen: dict[tuple[int, int], dict] = {}
    used = 0

    def add(h: dict) -> None:
        nonlocal used
        key = _key(h)
        if key not in chosen and len(chosen) < MAX_CHUNKS and used + len(h["text"]) <= budget:
            chosen[key] = h
            used += len(h["text"])

    for h in keyword[:KEYWORD_SLOTS]:
        add(h)
    need = [d for d in doc_ids if d not in {c[0] for c in chosen}]
    broad = bool(_BROAD.search(question))
    per_doc = [vectorstore.query(vector, 2 if broad else 1, doc_id=d) for d in need]
    for rank in range(2 if broad else 1):  # everyone's best chunk first, then second-best chunks
        layer = [chunks[rank] for chunks in per_doc if len(chunks) > rank]
        for h in sorted(layer, key=lambda h: h["score"], reverse=True):
            if broad or h["score"] >= MIN_SCORE:
                add(h)
                sem_keys.add(_key(h))
    for key in sorted(candidates, key=fused.get, reverse=True):
        if candidates[key]["score"] >= cutoff:
            add(candidates[key])

    hits = sorted(chosen.values(), key=lambda h: fused.get(_key(h), 0.0), reverse=True)
    for h in hits:
        h["via"] = "both" if _key(h) in sem_keys and _key(h) in kw_keys else "keyword" if _key(h) in kw_keys else "semantic"
    return hits


def build_messages(
    history: list[dict], question: str, hits: list[dict], docs: list[dict], docs_only: bool = True
) -> list[dict]:
    """history: prior {role, content} turns. Sources are attached to the latest question only."""
    system = SYSTEM_PROMPT.format(
        inventory=inventory(docs), rules=_RULES_STRICT if docs_only else _RULES_GENERAL
    )
    if hits:
        blocks = []
        for i, h in enumerate(hits, 1):
            where = f"{h['filename']}, p. {h['page']}" if h["page"] else h["filename"]
            blocks.append(f"[{i}] ({where})\n{h['text']}")
        user = "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    else:
        user = f"(No documents matched this question.)\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": user}]
