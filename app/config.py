"""Providers, model registry and RAG settings. Both providers expose OpenAI-compatible endpoints."""
import os

from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    "Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
    },
}

# Ids are provider-side model names. "extra" is merged into the request body.
# "context_chars" = max characters of document excerpts sent per question (~4 chars per token;
# Groq free tiers have tight per-minute token limits).
MODELS = [
    {"provider": "Gemini", "id": "gemini-3.6-flash", "label": "Gemini 3.6 Flash", "context_chars": 40000},
    {"provider": "Groq", "id": "openai/gpt-oss-20b", "label": "GPT-OSS 20B", "context_chars": 12000},
    {
        "provider": "Groq",
        "id": "qwen/qwen3.8-27b",
        "label": "Qwen3.8 27B",
        "context_chars": 12000,
        "extra": {"reasoning_format": "hidden"},  # keep <think> blocks out of answers
    },
]


def api_key(provider: str) -> str | None:
    return os.getenv(PROVIDERS[provider]["key_env"]) or None


def is_available(model_id: str) -> bool:
    m = get_model(model_id)
    return bool(m and api_key(m["provider"]))


def get_model(model_id: str) -> dict | None:
    return next((m for m in MODELS if m["id"] == model_id), None)


def available_models() -> list[dict]:
    return [m for m in MODELS if api_key(m["provider"])]


# First model with a configured key, else the first model (UI then shows a setup hint).
DEFAULT_MODEL = (available_models() or MODELS)[0]["id"]


def context_chars_for(model_id: str) -> int:
    m = get_model(model_id)
    return (m or {}).get("context_chars", CONTEXT_CHARS)


def model_label(model_id: str) -> str:
    m = get_model(model_id)
    return f"{m['provider']} · {m['label']}" if m else model_id


# ---------- RAG ----------
# Fixed on purpose: changing the embedding model invalidates every stored vector.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 150
CONTEXT_CHARS = 16000  # default excerpt budget when a model sets no "context_chars"
MAX_CHUNKS = 30  # hard cap on chunks per question, whatever the budget
REL_CUTOFF = 0.6  # drop chunks scoring below this fraction of the best match...
MIN_SCORE = 0.20  # ...or below this cosine similarity (also needed for a document's guaranteed chunk)
KEYWORD_SLOTS = 3  # best keyword matches always included, even when their cosine score is low
RRF_K = 60  # reciprocal rank fusion constant for merging semantic and keyword results
MAX_HISTORY_MESSAGES = 10  # prior chat messages sent to the LLM (5 exchanges)
