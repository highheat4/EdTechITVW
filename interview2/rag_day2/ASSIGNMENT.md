# 📘 Day 2 Assignment — Chunking & Embedding Ablation

## 🎯 Objective

Take your Day 1 baseline and:

1. Add multiple **chunking strategies**.
2. Add multiple **embedding models**.
3. Compare **configurations** using retrieval evaluation (precision\@k, latency).
4. Present a **tradeoff table** of results.

---

## 🛠️ Requirements

### 1. Multiple Chunkers

Implement at least 2 (ideally 3):

* **Fixed-length** (600 chars, 100 overlap — already done).
* **Sliding window** (e.g., 300 chars, stride 200).
* **Sentence/paragraph split** (split on `"."` or `"\n"` then group into \~400–600 char windows).

Each chunker should be a function you can pass into `load_docs`. Example:

```python
def chunk_fixed(text, size=600, overlap=100): ...
def chunk_sliding(text, size=300, stride=200): ...
def chunk_semantic(text, max_len=500): ...
```

### 2. Multiple Embedding Models

Test at least 2 models:

* `all-MiniLM-L6-v2` (fast, 384-dim).
* `gte-base` (better quality, 768-dim).
* *(optional)* `gte-large` if you want a “quality at cost” example.

Make your `Embedder` class take a model name argument.

### 3. Evaluation Harness

Extend Day 1’s `eval.py`:

* Input: `eval.jsonl` (queries + gold answers).
* Output:

  * **precision\@k (hit\@k)** = % of queries where a retrieved chunk contains the gold string.
  * **median retrieval latency (ms)**.

Add flags or a simple loop to evaluate:

* Different chunkers
* Different embedding models
* Different `k` values (e.g., 3, 5, 8)

### 4. Record Results

Make a simple results table like:

| Chunker          | Embedder     | k | p\@k | Median Latency (ms) |
| ---------------- | ------------ | - | ---- | ------------------- |
| fixed(600,100)   | MiniLM-L6-v2 | 5 | 0.80 | 7.5                 |
| sliding(300,200) | MiniLM-L6-v2 | 5 | 0.86 | 8.2                 |
| semantic(500)    | gte-base     | 5 | 0.92 | 15.4                |

*(Numbers will be yours — these are examples.)*

### 5. Write Down Tradeoffs

At the end of your run, add a short note:

* *“Sliding window improved recall but added 10–15% latency.”*
* *“gte-base embeddings gave higher p\@k but doubled latency.”*
* *“Default config chosen: fixed 600/100 + MiniLM (balance of speed + recall).”*

---

## ✅ Deliverables

* Updated retriever that supports different chunkers + embedding models.
* Updated `eval.py` that prints metrics.
* A results table (Markdown is fine).
* A short paragraph about tradeoffs and your default choice.

---

## 🌶️ Stretch Goals

* Try different FAISS index types:

  * `IndexFlatIP` (baseline).
  * `IndexHNSWFlat(d, 32)` (faster, approximate).
* Compare p\@k vs latency for Flat vs HNSW.

---

## 🧭 Success Criteria

* You can run `python eval.py` with different configs and get meaningful numbers.
* You have at least 1–2 insights about tradeoffs (not just raw numbers).
* You can **say out loud**: *“I chose X config because it gives 0.85 precision\@5 at 8 ms, which balances recall and speed for an interactive tutor.”*

---
