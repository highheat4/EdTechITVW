Day 3 — Context Engineering: Budgeting, Prompting, Refusal

🎯 Objective

Given your existing RAG pipeline, implement a token-aware context builder and prompt strategy that:

fits a fixed context budget (≤ 700 tokens) without losing key evidence,

deduplicates overlapping content,

refuses when evidence is weak,

shows a measurable win on an eval set (accuracy/precision@k and/or hallucination reduction),

logs latency and token counts.

Timebox: 2–3 hours.

What to Build

1) Token Budgeter

Create a function that accepts retrieved chunks (each has text, score, doc) and returns a compact context string under a token cap.

Requirements

Token counting: use tiktoken (or HF tokenizer) to count tokens.

Ordering: sort by score descending.

Deduping: drop near-duplicates (e.g., Jaccard > 0.9 or cosine sim > 0.95 with a cheap embedding) and same-doc repeats if they don’t add unique sentences.

Inclusion policy:

Add top chunks until adding another would exceed the budget.

If the next chunk overflows the budget but is high-score, include a summary of it (simple heuristic: first N sentences or a 1–2 sentence extract using sentence ranks).

Citations: prefix every chunk with [doc=..., score=...].

Signature

def build_context(chunks, max_tokens=700, dedupe=True, summarize_overflow=True) -> str:
    ...

2) Prompt Template + Refusal

Design a minimal but robust prompt that forces evidence-bound answers.

System prompt (example)

You are an edtech tutor. Answer ONLY using the provided context.
If the context is insufficient, reply exactly: "I don't know."
Cite the sources in parentheses like (doc.md).
Be concise (<= 3 sentences).

User prompt

Question: {query}

Context:
{context_from_budgeter}

Refusal rule

If top_score < τ OR total tokens in context < min_context_tokens, return "I don't know."

Default thresholds: τ ∈ [0.25, 0.35], min_context_tokens ≈ 80 (tune).

3) Logging & Telemetry

Log per query:

retrieval_ms, budgeting_ms, total_tokens, context_tokens, answer_tokens

top_score, num_chunks_included, num_deduped, num_summarized

a boolean refused

Evaluation (prove it helped)

Use the Day-1/2 tiny corpus or your larger set. Create/extend eval.jsonl with 15–30 items.

Metrics

Answer accuracy (simple): does the gold phrase appear in the answer? (string or semantic match)

Refusal correctness: fraction of deliberately out-of-scope queries refused.

Latency: p50/p90 for (retrieval + budgeting).

Tokens: mean total tokens passed to the LLM (proxy for cost).

Experiments (run both):

Baseline (concatenate top-k blindly, no budget, no refusal).

Context-engineered (your budgeter + refusal + citations).

Report (print a tiny table):

Setup                 Acc@N   Refusal@OOS   p50_latency_ms   mean_tokens
Baseline              0.64    0.10           95               1350
Context-engineered    0.78    0.86           118               680

(Numbers illustrative — show yours.)

Reference Skeleton (drop-in stubs)

# context.py
import time, math
import tiktoken
from typing import List, Dict

def token_len(text: str, enc=None):
    enc = enc or tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def near_duplicate(a: str, b: str, thresh=0.9):
    # quick Jaccard on word sets; good enough for small docs
    A, B = set(a.lower().split()), set(b.lower().split())
    j = len(A & B) / max(1, len(A | B))
    return j >= thresh

def summarize_heuristic(text: str, max_sent=2):
    # naive: first N sentences
    sents = [s.strip() for s in text.split('.') if s.strip()]
    return '. '.join(sents[:max_sent]) + ('.' if sents else '')

def build_context(chunks: List[Dict], max_tokens=700, dedupe=True, summarize_overflow=True):
    t0 = time.perf_counter()
    enc = tiktoken.get_encoding("cl100k_base")

    # sort by score desc
    chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)

    used, out, kept_texts = 0, [], []
    for c in chunks:
        header = f"[{c['doc']} | score={c['score']:.2f}]"
        body = c['text'].strip()

        if dedupe and any(near_duplicate(body, t) for t in kept_texts):
            continue

        candidate = f"{header}\n{body}\n"
        need = token_len(candidate, enc)
        if used + need <= max_tokens:
            out.append(candidate)
            kept_texts.append(body)
            used += need
            continue

        if summarize_overflow:
            summary = summarize_heuristic(body)
            cand2 = f"{header}\n{summary}\n"
            need2 = token_len(cand2, enc)
            if need2 <= max_tokens - used:
                out.append(cand2)
                used += need2
                continue

        # cannot fit anything more
        break

    return "\n".join(out), {"budgeting_ms": (time.perf_counter()-t0)*1000,
                            "tokens_used": used, "chunks_kept": len(out)}

# refusal.py
def should_refuse(top_score: float, context_tokens: int, score_thresh=0.3, min_ctx=80):
    return (top_score < score_thresh) or (context_tokens < min_ctx)

# qa_pipeline.py (fragment)
from context import build_context, token_len
from refusal import should_refuse

SYSTEM = ("You are an edtech tutor. Answer ONLY with the provided context. "
          "If insufficient, reply exactly: \"I don't know.\" Cite sources like (doc.md). "
          "Be concise (<= 3 sentences).")

def answer(query, retrieved, llm_call):
    # retrieved: [{doc, text, score}, ...]
    top = retrieved[0] if retrieved else {"score": 0.0}
    ctx, m = build_context(retrieved, max_tokens=700)
    if should_refuse(top.get("score",0.0), m["tokens_used"]):
        return {"answer": "I don't know.", "ctx": ctx, "meta": {"refused": True, **m}}

    prompt = f"{SYSTEM}\n\nQuestion: {query}\n\nContext:\n{ctx}\n"
    # plug your LLM or template-based extraction
    out = llm_call(prompt)  # or your deterministic stub during practice
    return {"answer": out, "ctx": ctx, "meta": {"refused": False, **m}}

Test Cases (include in your eval)

Add 5–10 out-of-scope queries to eval.jsonl, e.g.:

{"q":"Does Acme Tutor support Spanish translation?","gold":"OOS","doc":""}
{"q":"What is the CEO's email?","gold":"OOS","doc":""}

Scoring rule: an answer of "I don't know." counts as correct for OOS.

Success Criteria (Day 3)

Budgeter fits context ≤ 700 tokens with dedupe and optional summarization.

Refusal triggers correctly on OOS or weak evidence.

Measurable improvement over baseline in at least one axis:

↑ Answer accuracy (or ↓ hallucinations on OOS),

↓ Mean tokens (cost),

↔/↓ Latency (budgeting adds small overhead but stays reasonable).

Level Signals (what this round shows)

Dimension

MLE I

MLE II

Staff

Budgeting

Truncates raw concat

Token-aware include until cap

Dedupes + selective summarize; logs tokens saved

Prompting

Basic template

Clear system prompt + citations

Adds refusal guardrails + concise style; parameterized

Refusal

None

Threshold on top score

Threshold + min-context + returns explanations in meta

Eval

Manual

Acc@N + latency p50

Adds OOS set; reports tokens & tradeoff commentary

10-minute Wrap Script (for interview)

“I added a token-aware context budgeter that dedupes and summarizes overflow so we fit ≤700 tokens.A refusal policy triggers if evidence is weak.On the eval set, vs baseline, accuracy improved from X→Y and mean tokens dropped Z% with p50 latency ≈ N ms. Next I’d add a small cross-encoder reranker under a feature flag to boost precision when k grows.”