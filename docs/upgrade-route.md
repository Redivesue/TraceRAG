# Upgrade Route

## Goal
This document summarizes how **Perplexity Lite** evolved from a runnable offline QA demo into a faster, more grounded search-style QA system with real LLM integration, citation control, and fast-path inference.

## Phase 1: Data and Core Scaffold
Initial work focused on getting a stable project skeleton and a usable dataset split in place.

- Added project docs and architecture notes.
- Built the initial module layout under `src/` for retrieval, rerank, generation, pipeline, API, data, and evaluation.
- Downloaded and validated `HotpotQA`:
  - `train` for corpus building
  - `dev_distractor` for evaluation
- Exported paragraph-level chunks to `artifacts/chunks/train_chunks.jsonl`.

Outcome:
- The system had a correct `train -> corpus / dev -> eval` data split.
- The repo moved from “empty scaffold” to a runnable offline MVP.

## Phase 2: End-to-End Pipeline
The next step was making the full QA chain work.

- Implemented sparse, dense, and hybrid retrieval.
- Added a heuristic reranker.
- Added answer generation with citations.
- Exposed the pipeline through FastAPI and Streamlit.
- Added retrieval and citation evaluation scripts.

Outcome:
- The full pipeline ran:
  `query -> retrieve -> rerank -> generate -> cite`
- The project was no longer just a static demo.

## Phase 3: Real LLM Integration
The generation layer was then upgraded from offline-only fallback to a real provider-backed setup.

- Added an OpenAI-compatible client in [`openai_compatible.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/generation/openai_compatible.py).
- Added `.env`-driven provider config in [`config.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/core/config.py).
- Verified DeepSeek integration using:
  - `LLM_PROVIDER=openai_compatible`
  - `LLM_MODEL=deepseek-chat`
  - `LLM_BASE_URL=https://api.deepseek.com`

Outcome:
- Hosted LLM generation worked.
- The main bottleneck was confirmed to be retrieval quality, not provider integration.

## Phase 4: Quality Optimization
Once the system was connected end-to-end, the focus shifted to answer quality.

Main problems found:
- entity mismatch in retrieval
- repeated chunks from the same title
- weak coverage for comparison questions
- citations returning the whole top-k instead of only used evidence

Fixes:
- `bm25.py`: query decomposition, entity extraction, title/entity boosts, attribute hints
- `hybrid.py`: title-level dedup and entity coverage selection
- `hosted.py`: entity/title/attribute scoring, mismatch penalties, diversity-aware rerank
- `llm_generator.py`: comparison-oriented answer structure and citation compression

Outcome:
- Comparison questions like
  `Were Scott Derrickson and Ed Wood of the same nationality?`
  improved from incorrect or overly cautious output to grounded correct answers.

## Phase 5: Latency Profiling and Cold-Start Reduction
After answer quality improved, the next issue was latency.

Fixes:
- Added stage timings in [`search_pipeline.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/pipeline/search_pipeline.py):
  - retrieval
  - rerank
  - generation
  - total
- Added persisted retriever caches under `artifacts/indexes/`.
- Added in-process chunk caching in [`corpus.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/data/corpus.py).

Outcome:
- Retrieval was identified as the dominant slow path.
- Cold-start total latency dropped from about `20.45s` to about `13.76s`.

## Phase 6: Fast-Path Inference
To reach practical interactive latency, the system introduced a separate fast path for common entity-centric questions.

New components:
- [`title_fast.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/title_fast.py): lightweight exact-title retrieval
- [`lazy.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/lazy.py): lazy heavy-index loading

Fast-path optimizations:
- title-first retrieval before heavy sparse/dense loading
- query/result caches in retrieval and pipeline
- local rule-based answer generation for:
  - nationality
  - same nationality
  - profession
  - older / younger

Observed results for common entity questions:
- single-entity questions: sub-millisecond to low-millisecond total latency
- simple comparison questions: sub-millisecond total latency

Examples:
- `What nationality is Scott Derrickson?`
- `Were Scott Derrickson and Ed Wood of the same nationality?`
- `Who is older, Scott Derrickson or Ed Wood?`

## Phase 7: Fast-Path Expansion and Cache Compression
The next round focused on widening the fast path and reducing retriever cache size.

Fixes:
- Expanded fast-path coverage in [`llm_generator.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/generation/llm_generator.py) to include:
  - same profession
  - same birthplace
  - born in which city / town
  - older / younger
- Improved fast-path answer formatting for profession outputs.
- Compressed title cache into [`train_chunks.title.pkl.gz`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/artifacts/indexes/train_chunks.title.pkl.gz).
- Rebuilt BM25 persistence into compressed form [`train_chunks.bm25.pkl.gz`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/artifacts/indexes/train_chunks.bm25.pkl.gz) for measurement.

Observed results:
- `What is Ed Wood's profession?` now returns:
  `Ed Wood is an American filmmaker [1].`
- `Did Scott Derrickson and Ed Wood have the same profession?` now uses `fast_path`.
- `train_chunks.title.pkl.gz` reduced title-cache size to about `113MB`.
- `train_chunks.bm25.pkl.gz` reduced BM25 cache size from about `2.2GB` to about `460MB`.

Tradeoff:
- The compressed BM25 cache is much smaller, but the current serialized structure still loads too slowly for practical slow-path startup.
- This means fast-path latency is now excellent for supported question types, but slow-path retrieval still needs a more compact runtime representation.

## Phase 8: BM25 Runtime Redesign
The next step was a deeper structural optimization rather than another small cache tweak.

Main change:
- Reworked [`bm25.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/bm25.py) so the persisted runtime no longer stores a full per-document `Counter` list.
- The new cache keeps only the sparse state actually needed at query time:
  - postings with `(doc_id, term_freq)`
  - document frequency
  - document length
  - exact-title and title-token maps
- Query-time scoring now reconstructs only the local token frequencies needed for the active candidate set.

Related follow-up:
- Expanded birthplace extraction patterns in [`llm_generator.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/generation/llm_generator.py) for phrases like:
  - `born at ...`
  - `from ...`
  - parenthetical location forms such as `(X, Country)`
- Hardened [`dense.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/dense.py) so incompatible old cache payloads fall back to rebuild instead of crashing.

Observed BM25 benchmark:
- previous compressed BM25 cache:
  - size about `460MB`
  - load time about `158.58s`
- redesigned BM25 cache:
  - size about `127.63MB`
  - build time about `151.95s`
  - load time about `10.18s`

Outcome:
- The slow-path sparse retriever moved from “compressed but still impractical” to “small enough and fast enough for this iteration”.
- The biggest remaining slow-path risk is no longer BM25 structure itself, but other heavyweight components such as dense index load and hosted generation latency.

## Current State
The project now has two effective modes:

- `fast_path`: very low latency for common structured entity questions
- `openai_compatible`: hosted LLM-backed fallback for harder or unsupported queries

This makes the system closer to a production-style search QA design than a basic RAG demo.

Practical stopping point for this iteration:
- Fast-path structured entity questions should stay around millisecond-level latency.
- Slow-path questions can remain slower for now, as long as they are correct and clearly instrumented.
- BM25 cache load should stay in the low-seconds to ~10-second range rather than minute-level startup.
- Further optimization beyond this point should target dense retrieval redesign or a different indexing backend, not more prompt tweaking.

## Remaining Work
The most useful next upgrades are:

1. Redesign dense retrieval persistence so old or oversized caches do not dominate slow-path startup.
2. Improve birthplace extraction only where evidence explicitly contains city-level information.
3. Add benchmarks that compare:
   - fast path latency
   - cold slow path latency
   - warm slow path latency
4. Add resume-ready evaluation summaries with both quality and latency metrics.

## Phase 9: Two-Stage Retrieval Refactor
The next upgrade moved the retrieval stack toward a more standard search architecture.

Main changes:
- Added offline index building in [`build_index.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/indexing/build_index.py).
- Added query normalization in [`query_normalizer.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/query_normalizer.py).
- Replaced ad hoc hybrid score merging with:
  - reciprocal rank fusion in [`fusion.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/fusion.py)
  - title-level dedup in [`dedup.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/dedup.py)
- Reworked [`dense.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/dense.py) into an offline-built vector index loader.
- Kept the dense layer FAISS-ready: if `faiss` is available, it will load or write a FAISS index; otherwise it uses a local NumPy vector index fallback.

Generated offline assets:
- [`train_chunks.metadata.json`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/artifacts/indexes/train_chunks.metadata.json)
- [`train_chunks.vector.matrix.npy`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/artifacts/indexes/train_chunks.vector.matrix.npy)
- [`train_chunks.vector.meta.pkl`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/artifacts/indexes/train_chunks.vector.meta.pkl)

Observed results:
- cold slow-path retrieval for `What did the Pound–Rebka experiment test?` dropped from about `335s` to about `13-15s`
- warm slow-path retrieval for another query in the same process dropped to about `0.12s`
- fast-path structured entity questions remained around millisecond-level

Interpretation:
- The main retrieval bottleneck is no longer query-time corpus construction.
- The remaining cold-path delay is now dominated by first-load index hydration, especially BM25 load.
- This is a meaningful architectural improvement even though cold-start latency is still higher than the ideal target.

## Phase 10: FAISS Activation
The retrieval stack was then upgraded from FAISS-ready to actual FAISS usage.

Main changes:
- Installed `faiss-cpu` and updated project dependencies.
- Rebuilt the dense retrieval assets so the project now writes and loads:
  - [`train_chunks.vector.faiss.index`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/artifacts/indexes/train_chunks.vector.faiss.index)
  - [`train_chunks.vector.meta.pkl`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/artifacts/indexes/train_chunks.vector.meta.pkl)
- Updated runtime boot so BM25 and FAISS indexes are preloaded at service startup by default.
- Updated dense retrieval source labels to distinguish:
  - `dense_faiss`
  - `dense_numpy`

Outcome:
- The project now uses a real FAISS vector index in the dense retrieval path.
- Remaining retrieval latency is no longer blocked by query-time vector index construction.
- The main remaining latency issue is BM25 cold-load cost and the quality of the lightweight local embedding function, not the absence of a vector index backend.

## Phase 11: Dense Encoder Upgrade Path
The next quality-focused upgrade was to separate the dense retrieval backend from the vector store itself.

Main changes:
- Refactored [`dense.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/dense.py) so FAISS can be backed by different encoders instead of the old fixed hash embedding.
- Added a pluggable encoder layer in [`encoders.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/encoders.py).
- Added offline build options in [`build_index.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/indexing/build_index.py):
  - `--encoder-backend hash`
  - `--encoder-backend transformer`
  - `--model-name sentence-transformers/all-MiniLM-L6-v2`
- Added runtime config switches:
  - `DENSE_ENCODER_BACKEND`
  - `DENSE_MODEL_NAME`
  - `DENSE_ENCODE_BATCH_SIZE`
- Disabled the experimental query-time semantic refiner by default. The stable path is now: offline dense encoding -> FAISS search -> rerank.

Validation:
- On a toy corpus with one relevant document and two lexical distractors:
  - hash FAISS scores were:
    - `Pound–Rebka experiment 0.757`
    - `What Katy Did Next 0.4629`
    - `I Still Know What You Did Last Summer 0.4444`
  - transformer FAISS scores became:
    - `Pound–Rebka experiment 0.7856`
    - `I Still Know What You Did Last Summer 0.0904`
    - `What Katy Did Next 0.0589`

## Phase 12: Productization Traceability
The next upgrade focused on making runtime behavior easier to observe and explain.

Main changes:
- Added structured request logging in [`telemetry.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/core/telemetry.py).
- Extended pipeline responses with:
  - `request_id`
  - `trace.selected_path`
  - `trace.path_reason`
  - retrieval / rerank candidate counts
- Extended generation output so the pipeline can distinguish:
  - `fast_path_match`
  - `hosted_llm_success`
  - `llm_not_configured_offline_fallback`
  - `hosted_llm_error_offline_fallback`
  - `no_evidence`
- Added index manifest support in [`build_index.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/indexing/build_index.py).
- Added runtime metadata exposure in [`runtime.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/core/runtime.py) and `/health`.

Outcome:
- Requests are now traceable across API responses and logs.
- Fast-path and fallback behavior is easier to audit.
- Index state is easier to inspect without reading implementation code.

## Phase 12: Slow-Path Quality Hardening
After the transformer FAISS upgrade, the next bottleneck moved from speed to slow-path answer quality.

Main problems found:
- reranked tails still included duplicate or near-duplicate titles
- experiment-style questions still pulled some lexical-noise candidates
- offline fallback could repeat nearly identical evidence sentences

Fixes:
- tightened [`hosted.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/rerank/hosted.py) with:
  - stronger duplicate-title penalties
  - title-similarity filtering
  - lexical-noise penalties for weak overlaps
  - light redundancy penalties for repetitive chunks
- tightened [`llm_generator.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/generation/llm_generator.py) with:
  - normalized sentence dedup in offline fallback
  - suppression of repeated fallback snippets
- added fixed regression checks in:
  - [`regression_cases.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/evaluation/regression_cases.py)
  - [`regression.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/evaluation/regression.py)

Outcome:
- slow-path rerank results are less repetitive
- offline fallback answers are cleaner
- retrieval/generation changes now have a fixed smoke-test baseline instead of ad hoc manual checks

## Phase 13: Startup and UI Responsiveness
The next iteration focused on reducing perceived latency, especially around first-use slow-path queries and the Streamlit demo experience.

Main problems found:
- the first slow-path query still paid transformer query-encoder load time
- the Streamlit page looked frozen because it auto-ran a default query on initial render

Fixes:
- added `PREWARM_QUERY_ENCODER` in [`config.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/core/config.py)
- added dense query-encoder prewarm hooks in:
  - [`encoders.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/encoders.py)
  - [`dense.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/retrieval/dense.py)
  - [`runtime.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/core/runtime.py)
- changed the Streamlit app in [`streamlit_app.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/app/streamlit_app.py) to:
  - cache the pipeline resource
  - render the page before any search runs
  - require an explicit `Search` submit instead of auto-running a default query
- added benchmark reporting with latency percentiles in [`benchmark.py`](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/evaluation/benchmark.py)

Outcome:
- first slow-path model-load cost is moved closer to startup
- UI responsiveness is improved because initial page paint is no longer blocked by an automatic search
- the project now has both fixed regression checks and percentile-style latency benchmarking

Interpretation:
- FAISS was already solving the latency problem.
- The remaining retrieval noise came from the old hash embedding itself, not from the vector index backend.
- The project now has a clean upgrade path to rebuild dense indexes with a real retrieval model while keeping the same online retrieval architecture.
