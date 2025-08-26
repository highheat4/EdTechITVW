import os, glob, time
from chunking import chunk_fixed
from embedder import Embedder
from index import build_flat_index, search
from qa import compose_answer

# from qa import sear

def load_docs(doc_dir="./docs"):
    out = []
    for p in glob.glob(os.path.join(doc_dir, "*.md")):
        with open(p, "r", encoding = "utf-8") as f:
            text = f.read()
        for chunk in chunk_fixed(text):
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