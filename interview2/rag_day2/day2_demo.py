# eval.py
import argparse, json, time, statistics as stats
from pathlib import Path
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# Chunkers (list-returning)
# ----------------------------
def chunk_fixed(text: str, size: int = 600, overlap: int = 100):
    if not text:
        return []
    if size <= 0: raise ValueError("size must be > 0")
    if overlap >= size: raise ValueError("overlap must be < size")
    chunks, n, s = [], len(text), 0
    while True:
        e = min(s + size, n)
        chunks.append(text[s:e])
        if e >= n: break
        s = e - overlap
    return chunks

def chunk_sliding(text: str, size: int = 300, stride: int = 200):
    if not text:
        return []
    if size <= 0 or stride <= 0: raise ValueError("size/stride must be > 0")
    chunks, n, s = [], len(text), 0
    while True:
        e = min(s + size, n)
        chunks.append(text[s:e])
        if e >= n: break
        s += stride
    return chunks

def chunk_semantic(text: str, max_len: int = 500):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    chunks, buf = [], ""
    for sent in sentences:
        s = sent.strip()
        if not s: continue
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_len:
            buf += " " + s
        else:
            chunks.append(buf)
            buf = s
    if buf: chunks.append(buf)
    return chunks

CHUNKERS = {
    "fixed": chunk_fixed,
    "sliding": chunk_sliding,
    "semantic": chunk_semantic,
}

# ----------------------------
# Embedding wrapper
# ----------------------------
class Embedder:
    def __init__(self, name: str):
        self.name = name
        self.model = SentenceTransformer(name)
    def encode(self, texts):
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

# ----------------------------
# FAISS index helpers
# ----------------------------
def build_index(vecs: np.ndarray, index_type: str = "flat"):
    if vecs.ndim != 2 or vecs.shape[0] == 0:
        raise ValueError(f"Expected (n,d) embedding matrix, got {vecs.shape}")
    d = vecs.shape[1]
    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efSearch = 128
    else:
        index = faiss.IndexFlatIP(d)
    index.add(vecs)
    return index

def faiss_search(index, qvec: np.ndarray, k: int = 5):
    D, I = index.search(qvec.astype("float32"), k)
    return D[0].tolist(), I[0].tolist()

# ----------------------------
# Corpus loading
# ----------------------------
def load_docs(doc_dir: Path, chunker_name: str, **chunker_kwargs):
    chunker = CHUNKERS[chunker_name]
    docs = []
    md_paths = sorted(Path(doc_dir).glob("*.md"))
    if not md_paths:
        raise FileNotFoundError(f"No .md files found in {doc_dir}")
    for p in md_paths:
        text = p.read_text(encoding="utf-8")
        for ch in chunker(text, **chunker_kwargs):
            docs.append({"doc": p.name, "text": ch})
    return docs

# ----------------------------
# Retriever (self-contained)
# ----------------------------
class Retriever:
    def __init__(self, docs, embedder_name: str, index_type: str = "flat"):
        self.docs = docs
        self.embedder = Embedder(embedder_name)
        self.mat = self.embedder.encode([d["text"] for d in docs])
        self.index = build_index(self.mat, index_type=index_type)
    def retrieve(self, query: str, k: int = 5):
        qv = self.embedder.encode([query])
        t0 = time.perf_counter()
        scores, idxs = faiss_search(self.index, qv, k=k)
        ms = (time.perf_counter() - t0) * 1000.0
        results = [{"score": float(scores[i]), **self.docs[idxs[i]]} for i in range(len(idxs))]
        return results, ms

# ----------------------------
# Evaluation
# ----------------------------
def contains_gold(text: str, gold: str) -> bool:
    # case-insensitive containment; tweak if you need stricter matching
    return gold.lower() in text.lower()

def eval_config(doc_dir, eval_path, chunker_name, chunker_kwargs, embedder_name, k, index_type):
    docs = load_docs(doc_dir, chunker_name, **chunker_kwargs)
    r = Retriever(docs, embedder_name, index_type=index_type)
    latencies, hits = [], 0
    total = 0
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            res, ms = r.retrieve(ex["q"], k=k)
            latencies.append(ms)
            total += 1
            hit = any(contains_gold(rhit["text"], ex["gold_answer"]) for rhit in res)
            hits += int(hit)
    p_at_k = hits / total if total else 0.0
    median_ms = stats.median(latencies) if latencies else 0.0
    return {
        "chunker": chunker_name,
        "chunker_kwargs": chunker_kwargs,
        "embedder": embedder_name,
        "k": k,
        "index": index_type,
        "p_at_k": round(p_at_k, 3),
        "lat_ms_p50": round(median_ms, 1),
        "n_docs": len({d['doc'] for d in docs}),
        "n_chunks": len(docs),
    }

# ----------------------------
# Pretty output
# ----------------------------
def print_table(rows):
    if not rows:
        print("No results.")
        return
    headers = ["Chunker(config)", "Embedder", "k", "Index", "p@k", "lat ms p50", "docs", "chunks"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        ck = r["chunker"] + "(" + ",".join(f"{k}={v}" for k,v in r["chunker_kwargs"].items()) + ")"
        print("| " + " | ".join([
            ck,
            r["embedder"].split("/")[-1],
            str(r["k"]),
            r["index"].upper(),
            f"{r['p_at_k']:.3f}",
            f"{r['lat_ms_p50']:.1f}",
            str(r["n_docs"]),
            str(r["n_chunks"]),
        ]) + " |")
        
def make_table(rows):
    if not rows:
        return "No results."
    headers = ["Chunker(config)", "Embedder", "k", "Index", "p@k", "lat ms p50", "docs", "chunks"]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        ck = r["chunker"] + "(" + ",".join(f"{k}={v}" for k,v in r["chunker_kwargs"].items()) + ")"
        out.append("| " + " | ".join([
            ck,
            r["embedder"].split("/")[-1],
            str(r["k"]),
            r["index"].upper(),
            f"{r['p_at_k']:.3f}",
            f"{r['lat_ms_p50']:.1f}",
            str(r["n_docs"]),
            str(r["n_chunks"]),
        ]) + " |")
    return "\n".join(out)

def save_table(rows, path="results.md"):
    table = make_table(rows)
    with open(path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"[saved] {path}")

# ----------------------------
# Main (grid or single)
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc_dir", type=str, default="./interview2/docs")
    ap.add_argument("--eval_path", type=str, default="./interview2/eval.jsonl")
    ap.add_argument("--index", type=str, choices=["flat","hnsw"], default="flat")
    ap.add_argument("--grid", action="store_true", help="Run a small grid of configs")
    # single-run params (used if --grid is not set)
    ap.add_argument("--chunker", type=str, choices=list(CHUNKERS.keys()), default="fixed")
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=500)
    ap.add_argument("--embedder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    doc_dir = Path(args.doc_dir)
    eval_path = Path(args.eval_path)

    rows = []

    if args.grid:
        # ✱ Edit the grid below to your liking
        chunker_space = [
            ("fixed",   {"size": 600, "overlap": 100}),
            ("sliding", {"size": 300, "stride": 200}),
            ("semantic",{"max_len": 500}),
        ]
        embedder_space = [
            "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim, fast
            "thenlper/gte-base",                       # 768-dim, higher quality
        ]
        k_space = [3, 5, 8]

        for ch_name, ch_kwargs in chunker_space:
            for emb in embedder_space:
                for k in k_space:
                    res = eval_config(
                        doc_dir, eval_path,
                        ch_name, ch_kwargs,
                        emb, k, args.index
                    )
                    rows.append(res)
    else:
        # Single config
        if args.chunker == "fixed":
            ch_kwargs = {"size": args.size, "overlap": args.overlap}
        elif args.chunker == "sliding":
            ch_kwargs = {"size": args.size, "stride": args.stride}
        else:
            ch_kwargs = {"max_len": args.max_len}

        res = eval_config(
            doc_dir, eval_path,
            args.chunker, ch_kwargs,
            args.embedder, args.k, args.index
        )
        rows.append(res)

    print_table = make_table(rows)
    print(print_table)
    save_table(rows, "results.md")   # or change filename


if __name__ == "__main__":
    main()
