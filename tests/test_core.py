"""Офлайн-тесты: эмбеддинги и модель подменены заглушками, ключ не нужен."""
import hashlib

from src import config
from src.answer import ask
from src.cache import AnswerCache
from src.ingest import chunk_text, get_collection, ingest
from src.retriever import search


def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        out.append(v)
    return out


def make_index(tmp_path):
    col = get_collection(tmp_path / "chroma", reset=True)
    ingest(fake_embed, config.ROOT / "data" / "demo", collection=col)
    return col


def test_chunking_respects_size_and_min_length():
    text = "Первое предложение. " * 80
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(50 <= len(c) <= 260 for c in chunks)


def test_metadata_has_file_and_chunk(tmp_path):
    col = make_index(tmp_path)
    metas = col.get()["metadatas"]
    assert {m["source"] for m in metas} == {"FAQ.txt", "Регламент сервисной службы.txt"}
    assert all("chunk_id" in m for m in metas)


def test_source_filter(tmp_path):
    col = make_index(tmp_path)
    res = search(col, fake_embed, "время ответа на обращение", top_k=5, source="FAQ.txt")
    assert res and all(c.source == "FAQ.txt" for c in res)


def test_answer_has_citations_and_cache(tmp_path):
    col = make_index(tmp_path)
    cache = AnswerCache(tmp_path / "c.db")
    calls = []

    def llm(system, prompt):
        calls.append(prompt)
        return "ответ"

    r1 = ask("Сколько сессий на базовом тарифе?", col, fake_embed, llm, cache)
    r2 = ask("  сколько СЕССИЙ на базовом тарифе? ", col, fake_embed, llm, cache)
    assert not r1["from_cache"] and r2["from_cache"]
    assert len(calls) == 1
    assert r1["sources"] and {"file", "chunk", "text"} <= set(r1["sources"][0])
    assert "[Источник:" in calls[0]


def test_cache_key_depends_on_filter(tmp_path):
    cache = AnswerCache(tmp_path / "c.db")
    cache.set("вопрос", "a", [], source="FAQ.txt")
    assert cache.get("вопрос") is None
    assert cache.get("вопрос", "FAQ.txt")["answer"] == "a"
