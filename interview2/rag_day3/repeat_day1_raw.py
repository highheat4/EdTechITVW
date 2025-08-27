import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import json
from pathlib import Path
from typing import Callable, List, Dict, Tuple
import time
import statistics as stats


# chunking functions
def chunk_sliding(text: str, size: int = 300, stride: int = 200):
    chunks = []
    s = 0
    while s < len(text):
        if text[s:s+size].strip() != "":
            chunks.append(text[s:s+size].strip())
        s += stride
        # print(chunks)
    return chunks
    

def chunk_semantic(text: str, max_len: int = 500):
    chunks = []
    texts =  re.split(r'(?<=[.!?])\s+|\n+', text)
    for text in texts:
        if len(text) < max_len and text.strip() != "":
            chunks.append(text.strip())
        else:
            chunks += chunk_sliding(text, max_len, max_len)
        # print(chunks)
    return chunks

# embedding class
class Embedder:
    def __init__(self, name:str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)
    
    def encode(self,texts) -> np.ndarray:
        result = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(result, dtype="float32")
    
embedder = Embedder()


# indexing functions
def build_index(vecs, index_name:str = "faiss_index"):
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    return index

def search(query, index, k:int = 2):
    D, I = index.search(query, k)
    return D[0].tolist(), I[0].tolist()


def load_docs_chunked(doc_dir, chunker=chunk_semantic, **chunker_kwargs):
    out = []
    for p in Path(doc_dir).glob("*.md"):
        text = p.read_text(encoding="utf-8")
        for chunk in chunker(text, **chunker_kwargs):
            out.append({"doc": p.name, "text": chunk})
    return out


# evaluation utilities (kept small and modular)
def build_corpus_index(docs: List[Dict[str, str]]):
    texts = [d["text"] for d in docs]
    doc_names = [d["doc"] for d in docs]
    vecs = embedder.encode(texts)
    index = build_index(vecs)
    return index, texts, doc_names


def dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def read_jsonl(path: str) -> List[Dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def rank_in_list(target: str, items: List[str]):
    for i, it in enumerate(items, start=1):
        if it == target:
            return i
    return None


def contains_answer(texts: List[str], answer: str) -> bool:
    if not answer:
        return False
    a = str(answer).strip().lower()
    if not a:
        return False
    joined = " \n ".join(texts).lower()
    return a in joined


def compute_hit_mrr(ranks: List[int], k_values: Tuple[int, ...], total: int):
    hits = {k: 0 for k in k_values}
    mrr_sum = 0.0
    for r in ranks:
        if r is not None and r > 0:
            mrr_sum += 1.0 / r
            for k in k_values:
                if r <= k:
                    hits[k] += 1
    metrics = {f"hit@{k}": (hits[k] / total) for k in k_values}
    metrics["mrr"] = (mrr_sum / total) if total else 0.0
    return metrics


def evaluate_retrieval(
    eval_path: str,
    docs_dir: str,
    k_values: Tuple[int, ...] = (1, 3, 5),
    max_len: int = 500,
):
    # load and index docs
    docs = load_docs_chunked(docs_dir, chunker=chunk_semantic, max_len=max_len)
    if len(docs) == 0:
        raise ValueError(f"No markdown docs found under: {docs_dir}")
    index, corpus_texts, corpus_docs = build_corpus_index(docs)

    # read eval dataset
    eval_items = read_jsonl(eval_path)

    if len(eval_items) == 0:
        raise ValueError(f"Evaluation set is empty: {eval_path}")

    top_k = max(k_values)

    ranks: List[int] = []
    answer_found_at_k = {k: 0 for k in k_values}
    per_example: List[Dict] = []

    for ex in eval_items:
        q = ex.get("q") or ex.get("question")
        gold_doc = ex.get("gold_doc")
        gold_answer = ex.get("gold_answer", None)
        if not q or not gold_doc:
            continue

        q_vec = embedder.encode([q])
        scores, idxs = search(q_vec, index, k=top_k)
        retrieved_docs = [corpus_docs[i] for i in idxs]
        retrieved_texts = [corpus_texts[i] for i in idxs]

        # doc-level ranking by first appearance among chunks
        doc_ranking = dedupe(retrieved_docs)

        # compute rank and hits
        rank = rank_in_list(gold_doc, doc_ranking)
        ranks.append(rank)

        # answer string containment among top-k chunks (optional signal)
        if gold_answer:
            for k in k_values:
                if contains_answer(retrieved_texts[:k], gold_answer):
                    answer_found_at_k[k] += 1

        per_example.append({
            "q": q,
            "gold_doc": gold_doc,
            "top_chunks": [
                {"doc": corpus_docs[i], "score": scores[pos], "text": corpus_texts[i]}
                for pos, i in enumerate(idxs)
            ],
            "doc_ranking": doc_ranking,
            "rank": rank,
        })

    n = len(eval_items)
    metrics = compute_hit_mrr(ranks, k_values, n)
    if any(ex.get("gold_answer") for ex in eval_items):
        metrics.update({f"answer_found@{k}": answer_found_at_k[k] / n for k in k_values})

    return {"metrics": metrics, "examples": per_example}


def _print_summary(result):
    metrics = result["metrics"]
    print("Evaluation summary:")
    for k in sorted([int(k.split("@")[-1]) for k in metrics if k.startswith("hit@")]):
        key = f"hit@{k}"
        print(f"  {key}: {metrics[key]:.3f}")
    if "mrr" in metrics:
        print(f"  mrr: {metrics['mrr']:.3f}")
    for k in sorted([int(k.split("@")[-1]) for k in metrics if k.startswith("answer_found@")]):
        key = f"answer_found@{k}"
        print(f"  {key}: {metrics[key]:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval over docs using eval.jsonl")
    parser.add_argument("--eval_path", type=str, default="interview2/eval.jsonl", help="Path to eval.jsonl")
    parser.add_argument("--docs_dir", type=str, default="interview2/docs", help="Directory with .md docs")
    parser.add_argument("--k", type=int, nargs="*", default=[1, 3, 5], help="k values for Hit@k and answer_found@k")
    parser.add_argument("--max_len", type=int, default=500, help="Max chunk length for semantic splitter")
    args = parser.parse_args()

    res = evaluate_retrieval(
        eval_path=args.eval_path,
        docs_dir=args.docs_dir,
        k_values=tuple(args.k),
        max_len=args.max_len,
    )
    _print_summary(res)
