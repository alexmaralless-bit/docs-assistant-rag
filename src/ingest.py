"""Загрузка документов: чтение, нарезка на фрагменты, эмбеддинги, запись в ChromaDB."""
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings

from . import config
from .embeddings import EmbedFn

EXTENSIONS = (".txt", ".md")


def get_collection(path: Path | str | None = None, reset: bool = False):
    client = chromadb.PersistentClient(
        path=str(path or config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )
    if reset:
        try:
            client.delete_collection(config.COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})


def _overlap_tail(text: str, size: int) -> str:
    if size <= 0 or len(text) <= size:
        return text if size > 0 else ""
    tail = text[-size:]
    m = re.search(r"[.!?]\s+", tail)
    return tail[m.end():].strip() if m else tail.strip()


def _split_sentences(paragraph: str, size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > size:
            chunks.append(cur)
            tail = _overlap_tail(cur, overlap)
            cur = f"{tail} {s}".strip()
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP,
               min_len: int = config.MIN_CHUNK) -> list[str]:
    """Нарезка по абзацам; длинные абзацы режутся по предложениям; между фрагментами перекрытие."""
    chunks, cur = [], ""
    for par in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not par:
            continue
        if len(par) > size:
            if cur:
                chunks.append(cur)
                cur = ""
            parts = _split_sentences(par, size, overlap)
            chunks.extend(parts[:-1])
            cur = parts[-1]
        elif cur and len(cur) + len(par) + 2 > size:
            chunks.append(cur)
            tail = _overlap_tail(cur, overlap)
            cur = f"{tail}\n\n{par}" if tail else par
        else:
            cur = f"{cur}\n\n{par}" if cur else par
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) >= min_len]


def ingest(embed: EmbedFn, data_dir: Path | str | None = None, collection=None,
           reset: bool = True) -> int:
    """Индексирует все .txt/.md из папки. В метаданных: source (имя файла) и chunk_id."""
    data_dir = Path(data_dir or config.DATA_DIR)
    files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise SystemExit(f"В папке {data_dir} нет файлов .txt/.md")
    collection = collection or get_collection(reset=reset)
    docs, metas, ids = [], [], []
    for f in files:
        for i, chunk in enumerate(chunk_text(f.read_text(encoding="utf-8"))):
            docs.append(chunk)
            metas.append({"source": f.name, "chunk_id": i})
            ids.append(f"{f.stem}__{i}")
        print(f"{f.name}: обработан")
    vectors = []
    for start in range(0, len(docs), 64):
        vectors += embed(docs[start:start + 64])
    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
    print(f"Проиндексировано фрагментов: {len(docs)}")
    return len(docs)


if __name__ == "__main__":
    from .embeddings import openai_embedder

    ingest(openai_embedder())
