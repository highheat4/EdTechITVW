Here’s a **single, clean Markdown doc** you can paste directly into your repo as `ASSIGNMENT_DAY1.md` — no broken fences, all sections included.

# 📘 Day 1 Assignment — Minimal RAG QA System

## 🎯 Objective

Build a **minimal retrieval-augmented QA tool** over a small set of Markdown docs.
This is a from-scratch implementation: **no LangChain / LlamaIndex**.
Demonstrate an end-to-end pipeline that answers user queries with *citations* and *measured latency*.

---

## 📂 Dataset

Create a folder `./docs/` with **five** files and the following content:

**`product.md`**

```
# Acme Tutor Product
Acme Tutor is an AI teaching assistant focused on grades 6–12. It supports math, science, and history with step-by-step explanations and Socratic hints.
```

**`onboarding.md`**

```
# Onboarding
Students create an account with email or SSO. A diagnostic quiz sets initial mastery. The tutor adapts difficulty after each session based on mastery gains.
```

**`billing.md`**

```
# Billing
Two plans: Free and Pro. Pro costs $9.99 per month or $99 per year. Schools can request site licenses with volume discounts and teacher dashboards.
```

**`tutor_pedagogy.md`**

```
# Pedagogy
The tutor uses mastery-based progression and gives hints before revealing full solutions. It aligns to Common Core for math practice sets.
```

**`safety_privacy.md`**

```
# Safety & Privacy
PII is redacted prior to model calls. Student message logs are retained for 30 days. If the safety risk score exceeds 0.85, the session escalates to a human.
```

---

## 🛠️ Requirements

### 1) Document Loading & Chunking

* Implement `split_text(text, size=600, overlap=100)` that yields overlapping slices.
* Load all `.md` files from `./docs/` into a list of chunks with metadata:

  ```python
  {"doc": "product.md", "text": "...chunk..."}
  ```

**Reference snippet (optional):**

```python
def split_text(text: str, size: int = 600, overlap: int = 100):
    s, n = 0, len(text)
    while s < n:
        e = min(s + size, n)
        yield text[s:e]
        if e == n: break
        s = max(e - overlap, s + 1)
```

---

### 2) Embedding

* Use **sentence-transformers** (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
* Implement an `Embedder` with `encode(texts: List[str]) -> np.ndarray`.
* Normalize vectors so **inner product ≈ cosine similarity**.

**Reference snippet (optional):**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)
    def encode(self, texts):
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")
```

---

### 3) Indexing

* Use **FAISS FlatIP** index for the baseline.
* Build the index over all chunk embeddings.
* Must support:

  ```python
  scores, indices = search(index, qvec, k=5)
  ```

**Reference snippet (optional):**

```python
import faiss, numpy as np

def build_flat_index(vecs: np.ndarray):
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # vectors already normalized
    index.add(vecs)
    return index

def search(index, qvec: np.ndarray, k: int = 5):
    D, I = index.search(qvec, k)
    return D[0].tolist(), I[0].tolist()
```

---

### 4) Retrieval

* Implement `Retriever.retrieve(query: str, k=5)`:

  * Embed query → search FAISS → return **top-k** chunks with scores.
  * Record **retrieval latency (ms)** using `time.perf_counter()`.

**Reference wrapper (optional):**

```python
import os, glob, time

def load_docs(doc_dir="./docs"):
    out = []
    for p in glob.glob(os.path.join(doc_dir, "*.md")):
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        for chunk in split_text(text):
            out.append({"doc": os.path.basename(p), "text": chunk})
    return out

class Retriever:
    def __init__(self, doc_dir="./docs", model="sentence-transformers/all-MiniLM-L6-v2"):
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
```

---

### 5) QA Composition

* Implement `compose_answer(query, retrieved_chunks)`:

  * If **no chunks** or top score < `CONFIDENCE_T=0.25` → return `"I don't know."`
  * Else: return a concise answer **with citations** (top 1–2 doc names + scores).

**Reference snippet (optional):**

```python
CONFIDENCE_T = 0.25

def compose_answer(query, hits):
    top = hits[0] if hits else None
    if not top or top["score"] < CONFIDENCE_T:
        return "I don't know.", []
    cites = [(h["doc"], round(h["score"], 2)) for h in hits[:2]]
    # Minimal extractive answer for Day 1: cite the source(s)
    return f"Answer based on {cites[0][0]}. See citations.", cites
```

---

### 6) Demo CLI

* Create `run_demo.py`:

  * Loop: read user question → retrieve → print:

    * **Retrieval latency (ms)**
    * **Citations** (top-2)
    * **Answer string**

**Reference skeleton (optional):**

```python
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
```

---

### 7) Evaluation (bonus but recommended)

Create `eval.jsonl` in the project root:

```
{"q":"What grades does Acme Tutor focus on?","gold":"grades 6–12","doc":"product.md"}
{"q":"What is the monthly price of Pro?","gold":"$9.99 per month","doc":"billing.md"}
{"q":"How long are student messages retained?","gold":"30 days","doc":"safety_privacy.md"}
{"q":"What happens if risk score is high?","gold":"escalates to a human","doc":"safety_privacy.md"}
{"q":"How does the tutor adapt difficulty?","gold":"based on mastery gains","doc":"onboarding.md"}
```

Implement `eval.py` that:

* Runs retrieval for each query.
* Computes **hit\@k** (string containment of `gold` in any retrieved chunk).
* Prints **hit\@k** and **median retrieval latency (ms)**.

---

## ✅ Deliverables

* `python run_demo.py` runs end-to-end from a clean env.
* A retriever class that can be imported and tested.
* Console output includes **retrieval latency** and **citations** per query.
* *(Optional)* `eval.py` prints **hit\@k** and **median latency**.

---

## 🧭 Success Criteria

* End-to-end system works with **no external frameworks**.
* CLI answers queries over the toy corpus.
* On the sample eval set, **hit\@k ≥ 0.6** (≥3/5) for Day 1 is acceptable.
* Retrieval latency is printed for each query.

---

## 🌶️ Stretch Goals

* Swap FAISS **FlatIP → HNSW** and compare latency/accuracy (tune `efSearch`, `M`).
* Add a **token budgeter**: cap total context ≤ **700 tokens** (truncate extras).
* Print a short **tradeoff summary**: `k`, median latency, hit\@k.

---

## 🗣️ Interview Framing (how to present this work)

* “**Baseline RAG**: chunk → embed (MiniLM) → FAISS Flat → top-k → answer with **citations** and **refusal** when top score < 0.25.”
* “I **measure latency** at retrieval time and can report median across queries.”
* “**Next improvement**: switch to HNSW for lower latency, or add a simple **context budgeter** to keep tokens under budget.”

---

## ✅ Day 1 Checklist (copy into your TODOs)

* [ ] Create `./docs` with 5 files (content above).
* [ ] Implement `split_text(size=600, overlap=100)`.
* [ ] Load docs → list of `{doc, text}` chunks.
* [ ] Build embeddings with sentence-transformers (normalized).
* [ ] Build FAISS **FlatIP** index; verify `search()` returns scores/indices.
* [ ] Implement `Retriever.retrieve()` (top-k + **latency ms**).
* [ ] Implement `compose_answer()` with **citations** + **refusal** rule.
* [ ] Create `run_demo.py` CLI and test a few questions.
* [ ] *(Bonus)* Add `eval.jsonl` + `eval.py` (hit\@k + median latency).
* [ ] *(Stretch)* Try HNSW or a simple token budgeter and note tradeoffs.
