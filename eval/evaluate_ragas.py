"""Оценка качества ответов через RAGAS на наборе вопросов из eval/questions.json.

Запуск (нужны requirements-eval.txt и OPENAI_API_KEY): python -m eval.evaluate_ragas
"""
import json
from pathlib import Path

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness

from src import config
from src.answer import ask, make_llm
from src.embeddings import openai_embedder
from src.ingest import get_collection

QUESTIONS = Path(__file__).with_name("questions.json")


def build_dataset(top_k: int) -> Dataset:
    embed, llm, collection = openai_embedder(), make_llm(), get_collection()
    if collection.count() == 0:
        raise SystemExit("Индекс пуст: сначала python -m src.ingest")
    rows = {"user_input": [], "response": [], "retrieved_contexts": []}
    for item in json.loads(QUESTIONS.read_text(encoding="utf-8")):
        r = ask(item["question"], collection, embed, llm, cache=None, top_k=top_k)
        rows["user_input"].append(item["question"])
        rows["response"].append(r["answer"])
        rows["retrieved_contexts"].append([s["text"] for s in r["sources"]])
    return Dataset.from_dict(rows)


def main(top_k: int = config.TOP_K) -> None:
    dataset = build_dataset(top_k)
    llm = LangchainLLMWrapper(ChatOpenAI(model=config.CHAT_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=config.EMBEDDING_MODEL))
    result = evaluate(dataset, metrics=[Faithfulness(llm=llm),
                                        AnswerRelevancy(llm=llm, embeddings=emb),
                                        ContextPrecision(llm=llm)])
    print(result.to_pandas().to_string())


if __name__ == "__main__":
    main()
