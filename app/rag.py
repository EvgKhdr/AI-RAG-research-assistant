"""Retrieval-augmented prompting: fetch relevant chunks and build the LLM messages."""
import embeddings
import vectorstore
from config import TOP_K

SYSTEM_PROMPT = """You are a research assistant that answers questions using the user's documents.
Relevant excerpts are provided as numbered sources [1], [2], ... in the user's latest message.
- Base your answer on the sources and cite them inline like [1] or [2][3].
- If the sources do not contain the answer, say so plainly instead of guessing; you may then add general knowledge, clearly labelled as not coming from the documents.
- Do not invent citations. Be concise."""


def retrieve(question: str, folder_ids: list[int] | None = None, k: int = TOP_K) -> list[dict]:
    """Top-k chunks for the question; folder_ids=None searches every folder."""
    return vectorstore.query(embeddings.embed([question])[0], k, folder_ids)


def build_messages(history: list[dict], question: str, hits: list[dict]) -> list[dict]:
    """history: prior {role, content} turns. Sources are attached to the latest question only."""
    if hits:
        blocks = []
        for i, h in enumerate(hits, 1):
            where = f"{h['filename']}, p. {h['page']}" if h["page"] else h["filename"]
            blocks.append(f"[{i}] ({where})\n{h['text']}")
        user = "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    else:
        user = f"(No documents matched this question.)\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user}]
