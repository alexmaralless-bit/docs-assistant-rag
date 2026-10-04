"""Консольный ассистент: python -m src.cli [--source "FAQ.txt"] [--top-k 3]."""
import argparse

from . import config
from .answer import ask, make_llm
from .cache import AnswerCache
from .embeddings import openai_embedder
from .ingest import get_collection, ingest


def main() -> None:
    p = argparse.ArgumentParser(description="Ассистент по локальным документам")
    p.add_argument("--source", help="искать только в этом файле (имя файла)")
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    p.add_argument("--reindex", action="store_true", help="пересоздать индекс")
    args = p.parse_args()

    embed, llm, cache = openai_embedder(), make_llm(), AnswerCache()
    collection = get_collection()
    if args.reindex or collection.count() == 0:
        ingest(embed, collection=get_collection(reset=True))
        collection = get_collection()

    print("Вопрос в консоль. Команды: stats, clear (очистить кеш), exit")
    while True:
        try:
            q = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            break
        if q.lower() == "stats":
            print(f"Фрагментов: {collection.count()}; кеш: {cache.stats()}")
            continue
        if q.lower() == "clear":
            cache.clear()
            print("Кеш очищен")
            continue
        r = ask(q, collection, embed, llm, cache, args.top_k, args.source)
        print(f"\n{r['answer']}" + ("\n(из кеша)" if r["from_cache"] else ""))
        for s in r["sources"]:
            print(f"  - {s['file']}, фрагмент {s['chunk']}: {s['text'][:100]}...")


if __name__ == "__main__":
    main()
