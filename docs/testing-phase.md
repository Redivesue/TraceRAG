# Testing Phase

## Purpose
This phase validates the production-style retrieval pipeline after the dense backend upgrade from local hash vectors to transformer embeddings stored in FAISS.

## Current Runtime Baseline
- Dense backend: `transformer`
- Dense model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS
- Sparse retriever: compact BM25
- Fusion: reciprocal rank fusion (RRF)
- Dedup: title-level filtering before rerank
- Generator: DeepSeek through the OpenAI-compatible client, with offline fallback

## What To Verify
1. Retrieval quality improves beyond top-1, especially for experiment or concept questions.
2. End-to-end latency stays within an acceptable demo range after startup preload.
3. Citations only include chunks actually used by the final answer.
4. Fast-path entity questions still stay quick and correct.
5. Slow-path rerank results avoid duplicate or near-duplicate titles.
6. Offline fallback answers do not repeat the same sentence twice.

## Recommended Query Groups

### Fast Path
- `What nationality is Scott Derrickson?`
- `What is Ed Wood's profession?`
- `Who is older, Scott Derrickson or Ed Wood?`

### Slow Path Retrieval Quality
- `What did the Pound–Rebka experiment test?`
- `What is the purpose of the Nucifer experiment?`
- `What is general relativity about?`

### Failure / Guardrail Checks
- `What city was Christopher Nolan born in?`
- `Did Albert Einstein and Ed Wood work in the same field?`

## Success Criteria
- Top retrieved titles should be semantically tighter than the old hash baseline.
- The correct title should remain at or near rank 1.
- Dense-only noise should decrease for experiment-style questions.
- End-to-end answers should remain grounded and concise.
- Reranked evidence should not contain obvious same-title spam or near-duplicate titles.
- Offline fallback should avoid duplicated evidence sentences.

## Regression Runner
Run the fixed regression suite after retrieval or generation changes:

```bash
python3 -m src.evaluation.regression --debug
```

The regression suite covers:
- fast-path entity questions
- slow-path experiment / concept questions
- guardrail cases that should stay conservative

## Unified Evaluation Report
Run the unified report to aggregate retrieval quality, grounding, fast-path usage, bucketed regression behavior, and latency:

```bash
python3 -m src.evaluation.report --eval-limit 20 --benchmark-rounds 1
```

The report writes:
- `artifacts/reports/latest_report.json`
- `artifacts/reports/latest_report.md`
- for fast-path scope and fallback behavior, see `docs/fast-path-boundary.md`

Current report fields include:
- `Recall@5`
- `Recall@10`
- `MRR`
- citation hit rate
- citation precision
- answer grounded rate
- fast-path hit rate
- bucket-level pass rate and latency
- stage latency summary

## Notes
- First startup is still heavier because retrievers preload BM25 and FAISS into memory.
- Transformer FAISS improves retrieval quality, but startup and dense initialization cost are higher than the old hash baseline.
- The Streamlit UI now avoids auto-running the default query on first paint. The page should render first, then wait for an explicit `Search` click.
- `PREWARM_QUERY_ENCODER=true` moves the first transformer query-encoder load to service startup instead of the first slow-path question.
- `answer_grounded_rate` is currently an overlap-based proxy against HotpotQA supporting-fact titles, not a sentence-level entailment metric.

## Benchmark Runner
For a more formal latency summary with percentiles:

```bash
python3 -m src.evaluation.benchmark --warmup --rounds 3
```

This reports:
- mean latency
- p50 latency
- p95 latency
- max latency
