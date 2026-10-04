"""Эмбеддинги через OpenAI. Функцию можно подменить (в тестах)."""
from typing import Callable

from . import config

EmbedFn = Callable[[list[str]], list[list[float]]]


def openai_embedder() -> EmbedFn:
    from openai import OpenAI

    client = OpenAI(api_key=config.require_api_key())

    def embed(texts: list[str]) -> list[list[float]]:
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    return embed
