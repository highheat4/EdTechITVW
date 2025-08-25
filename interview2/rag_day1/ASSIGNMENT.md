# 📘 Day 1 Assignment — Minimal RAG QA System

## 🎯 Objective
Build a **minimal retrieval-augmented QA tool** over a small set of Markdown docs.  
This is a from-scratch implementation: **no LangChain / LlamaIndex**.  
You should demonstrate an end-to-end pipeline that can answer user queries with citations and measured latency.

---

## 📂 Dataset
Create a folder `./docs/` with 5 files:

- `product.md` — product overview (grades 6–12, subjects).
- `onboarding.md` — onboarding & diagnostic quiz.
- `billing.md` — free vs pro pricing.
- `tutor_pedagogy.md` — mastery-based pedagogy.
- `safety_privacy.md` — retention, PII, escalation.

*(Content provided in problem statement — copy/paste as-is.)*

---

## 🛠️ Requirements

### 1. Document Loading & Chunking
- Implement `split_text(text, size=600, overlap=100)`:
  - Yields overlapping slices of text.
  - Example: size 600, overlap 100 → each chunk is 600 chars, overlaps 100 with previous.
- Load all `.md` files from `./docs/` into a list of chunks:  
  ```python
  {"doc": "product.md", "text": "...chunk..."}
