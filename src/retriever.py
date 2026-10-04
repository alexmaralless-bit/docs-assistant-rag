"""Поиск фрагментов в ChromaDB с необязательным фильтром по источнику."""
from dataclasses import dataclass

from . import config
from .embeddings import EmbedFn


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    distance: float


def search(collection, embed: EmbedFn, query: str, top_k: int = config.TOP_K,
           source: str | None = None) -> list[Chunk]:
    """source: имя файла; фильтр применяется в базе до поиска (where), а не после."""
    total = collection.count()
    if total == 0:
        return []
    params = {"query_embeddings": [embed([query])[0]], "n_results": min(top_k, total)}
    if source:
        params["where"] = {"source": source}
    res = collection.query(**params)
    return [
        Chunk(text=d, source=m["source"], chunk_id=m["chunk_id"], distance=dist)
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
