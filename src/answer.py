"""Ответ по найденным фрагментам со ссылками на файл и номер фрагмента."""
from . import config
from .cache import AnswerCache
from .embeddings import EmbedFn
from .retriever import Chunk, search

SYSTEM = (
    "Ты помощник по внутренним документам. Отвечай только по приведённому контексту, "
    "по-русски, кратко. Если в контексте нет ответа, прямо скажи, что в документах "
    "этой информации нет. Ничего не выдумывай."
)


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    ctx = "\n\n".join(f"[Источник: {c.source}, фрагмент {c.chunk_id}]\n{c.text}" for c in chunks)
    return f"Контекст:\n{ctx}\n\nВопрос: {query}\n\nОтвет:"


def sources_of(chunks: list[Chunk]) -> list[dict]:
    return [{"file": c.source, "chunk": c.chunk_id, "text": c.text} for c in chunks]


def make_llm():
    from openai import OpenAI

    client = OpenAI(api_key=config.require_api_key())

    def llm(system: str, prompt: str) -> str:
        r = client.chat.completions.create(
            model=config.CHAT_MODEL, temperature=0.2, max_tokens=500,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()

    return llm


def ask(query: str, collection, embed: EmbedFn, llm, cache: AnswerCache | None = None,
        top_k: int = config.TOP_K, source: str | None = None) -> dict:
    """Кеш -> поиск -> промпт -> модель -> кеш. Возвращает ответ и список источников."""
    if cache:
        hit = cache.get(query, source, top_k)
        if hit:
            return {**hit, "from_cache": True}
    chunks = search(collection, embed, query, top_k, source)
    if not chunks:
        return {"answer": "В документах нет подходящей информации.", "sources": [], "from_cache": False}
    text = llm(SYSTEM, build_prompt(query, chunks))
    sources = sources_of(chunks)
    if cache:
        cache.set(query, text, sources, source, top_k)
    return {"answer": text, "sources": sources, "from_cache": False}
