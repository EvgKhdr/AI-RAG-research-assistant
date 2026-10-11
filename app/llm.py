from collections.abc import Iterator

from openai import OpenAI

from config import PROVIDERS, api_key, get_model

_clients: dict[str, OpenAI] = {}


def _client(provider: str) -> OpenAI:
    if provider not in _clients:
        key = api_key(provider)
        if not key:
            raise RuntimeError(f"{PROVIDERS[provider]['key_env']} is not set")
        _clients[provider] = OpenAI(api_key=key, base_url=PROVIDERS[provider]["base_url"])
    return _clients[provider]


def stream_chat(model_id: str, messages: list[dict]) -> Iterator[str]:
    """Yield answer text chunks. `messages` are OpenAI-style {role, content} dicts."""
    model = get_model(model_id)
    if model is None:
        raise ValueError(f"Unknown model: {model_id}")
    stream = _client(model["provider"]).chat.completions.create(
        model=model_id,
        messages=messages,
        stream=True,
        extra_body=model.get("extra"),
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
