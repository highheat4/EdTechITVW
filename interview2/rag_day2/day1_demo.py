def split_text(text, size = 600, overlap = 100):
    s = 0
    while s < max(1, len(text)):
        yield text[s:s+size]
        if s + size >= len(text): break
        s += size - overlap

from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, name="all-MiniLM-L6-v2"):
       self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def encode(self, texts):
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32")

import faiss, numpy as np

def build_flat_index(vecs: np.ndarray):
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    return index

def search(index, qvec: np.ndarray, k: int = 5):
    D, I  = index.search(qvec, k)
    return D[0].tolist(), I[0].tolist()

CONFIDENCE_T = 0.25

def compose_answer(query, hits):
    top = hits[0] if hits else None
    if not top or top["score"] < CONFIDENCE_T:
        return "I don't know.", []
    cites = [(h["doc"], round(h["score"], 2)) for h in hits[:2]]
    return f"Answer based on {cites[0][0]}. See citations.", cites

import time, glob, os

def load_docs(doc_dir="./docs"):
    out = []
    for p in glob.glob(os.path.join(doc_dir, "*.md")):
        with open(p, "r", encoding = "utf-8") as f:
            text = f.read()
        for chunk in split_text(text):
            out.append({"doc": os.path.basename(p), "text": chunk})
    return out


class Retriever:
    def __init__(self, doc_dir="./docs", model="all-MiniLM-L6-v2"):
        self.chunks = load_docs(doc_dir)
        self.embedder = Embedder(model)
        self.mat = self.embedder.encode([c["text"] for c in self.chunks])
        self.index = build_flat_index(self.mat)
        
    def retrieve(self, query, k=5):
        qv = self.embedder.encode([query])
        t0 = time.perf_counter()
        scores, idxs = search(self.index, qv, k)
        ms = (time.perf_counter() - t0) * 1000
        results = [{"score": float(scores[i]), **self.chunks[idxs[i]]} for i in range(len(idxs))]
        return results, ms
    
if __name__ == "__main__":
    r = Retriever("./docs")
    while True:
        q = input("\nAsk: ").strip()
        if not q: break
        hits, ms = r.retrieve(q, k=5)
        ans, cites = compose_answer(q, hits)
        print(f"Retrieval latency: {ms:.1f} ms")
        print(f"Citations: {cites}")
        print(ans)