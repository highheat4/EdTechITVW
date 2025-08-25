📘 Day 1 Assignment — Minimal RAG QA System
🎯 Objective

Build a minimal retrieval-augmented QA tool over a small set of Markdown docs.
This is a from-scratch implementation: no LangChain / LlamaIndex.
Demonstrate an end-to-end pipeline that answers user queries with citations and measured latency.

📂 Dataset

Create a folder ./docs/ with these 5 files and contents:

product.md

# Acme Tutor Product
Acme Tutor is an AI teaching assistant focused on grades 6–12. It supports math, science, and history with step-by-step explanations and Socratic hints.


onboarding.md

# Onboarding
Students create an account with email or SSO. A diagnostic quiz sets initial mastery. The tutor adapts difficulty after each session based on mastery gains.


billing.md

# Billing
Two plans: Free and Pro. Pro costs $9.99 per month or $99 per year. Schools can request site licenses with volume discounts and teacher dashboards.


tutor_pedagogy.md

# Pedagogy
The tutor uses mastery-based progression and gives hints before revealing full solutions. It aligns to Common Core for math practice sets.


safety_privacy.md

# Safety & Privacy
PII is redacted prior to model calls. Student message logs are retained for 30 days. If the safety risk score exceeds 0.85, the session escalates to a human.


(Optional) Tiny eval set in eval.jsonl:

{"q":"What grades does Acme Tutor focus on?","gold":"grades 6–12","doc":"product.md"}
{"q":"What is the monthly price of Pro?","gold":"$9.99 per month","doc":"billing.md"}
{"q":"How long are student messages retained?","gold":"30 days","doc":"safety_privacy.md"}
{"q":"What happens if risk score is high?","gold":"escalates to a human","doc":"safety_privacy.md"}
{"q":"How does the tutor adapt difficulty?","gold":"based on mastery gains","doc":"onboarding.md"}

🛠️ Requirements
1) Document Loading & Chunking

Implement split_text(text, size=600, overlap=100) to yield overlapping slices.

Load all .md files from ./docs/ into a list of chunk dicts:

{"doc": "product.md", "text": "...chunk..."}

2) Embedding

Use sentence-transformers (e.g., sentence-transformers/all-MiniLM-L6-v2).

Implement an Embedder class with:

encode(texts: List[str]) -> np.ndarray

Normalize vectors so inner product ≈ cosine similarity.

3) Indexing

Use FAISS IndexFlatIP for the baseline.

Build the index over all chunk embeddings.

Support a simple search API:

scores, indices = search(index, qvec, k=5)

4) Retrieval

Retriever.retrieve(query: str, k=5) should:

Embed the query, search FAISS, return top-k chunks with scores.

Record retrieval latency (ms) using a high-resolution timer.

5) QA Composition

Implement compose_answer(query, retrieved_chunks):

If no chunks or top score < CONFIDENCE_T = 0.25, return "I don't know."

Else return a concise 1–2 sentence answer and always include citations
(e.g., [('billing.md', 0.72)] showing top doc + score).

6) Demo CLI

Provide run_demo.py:

Interactive loop: read question → retrieve → print:

Retrieval latency (ms)

Citations (top-2 docs)

Answer string

7) Evaluation (bonus, recommended)

Implement eval.py over eval.jsonl:

For each query, run retrieval, compute hit@k
(does gold appear in any retrieved chunk via simple substring?).

Print hit@k and median retrieval latency (ms).

✅ Deliverables

Runnable demo: python run_demo.py

Importable Retriever class (and supporting functions/classes).

Console output shows retrieval latency and citations for each query.

(Bonus) python eval.py prints hit@k and median latency.

🧭 Success Criteria

End-to-end works with no external RAG frameworks.

Answers are grounded in retrieved text and include citations.

On the tiny eval set, hit@k ≥ 0.6 (≥3/5) is acceptable for Day 1.

Retrieval latency is printed per query.

🌶️ Stretch Goals

Swap FAISS FlatIP → HNSW and compare latency/precision.

Add a token budgeter: cap total context ≤ 700 tokens (truncate extra chunks).

Print a short tradeoff summary: k, median latency, hit@k.

🗣️ Interview Framing (how to present)

“Baseline RAG = chunk → embed → FAISS → retrieve → answer with citations.”

Show latency numbers and one or two example answers with sources.

Mention the next improvement you’d try (e.g., HNSW, reranker, or token budgeter).

✅ Checklist (copy into your TODO)

 Create ./docs/ with 5 files (contents above).

 Implement split_text(size=600, overlap=100).

 Build Embedder.encode() with normalized vectors.

 Build FAISS FlatIP index over all chunks.

 Implement Retriever.retrieve(query, k=5) + latency logging.

 Implement compose_answer() with citations and refusal rule.

 Create run_demo.py (interactive; prints latency + citations + answer).

 (Bonus) Add eval.py to report hit@k and median latency.

 (Stretch) Try HNSW and/or a token budgeter; note the tradeoff.