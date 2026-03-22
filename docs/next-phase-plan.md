# Next Phase Plan

## Goal
This phase does not change the core architecture again. The goal is to move the project from a strong prototype to a more review-ready, product-style system by improving:

1. metric clarity
2. problem-type coverage reporting
3. fast-path boundary definition
4. productization and observability

The project already has:
- offline-built BM25 + FAISS indexes
- transformer dense retrieval
- RRF fusion
- title-level dedup
- rerank
- citation-grounded answers
- regression and benchmark scripts

The next phase should focus on proving quality and making behavior easier to operate and explain.

## Review-Driven Priorities

### P0: Unified Evaluation Report
Add a single report script that summarizes:
- `Recall@5`
- `Recall@10`
- `MRR`
- citation precision
- answer grounded rate
- fast-path hit rate
- stage latency: retrieve / rerank / generate / total

Deliverables:
- `src/evaluation/report.py`
- `artifacts/reports/latest_report.json`
- `artifacts/reports/latest_report.md`

Why this matters:
- This turns “the system seems better” into “the system is measurably better”.
- It directly addresses the current evaluation gap noted in review.

Estimated effort:
- implementation: `0.5 to 1.5 days`
- compute cost on Mac M2: low
- GPU required: no

### P0: Problem-Type Bucketing
Split the regression and benchmark cases into labeled buckets:
- `nationality`
- `profession`
- `comparison`
- `concept_experiment`
- `guardrail`

For each bucket, report:
- accuracy or pass rate
- grounded rate
- fast-path hit ratio
- average latency

Deliverables:
- `src/evaluation/case_labels.py` or extension to existing regression cases
- bucketed report output in `report.py`

Why this matters:
- It makes system strengths and weaknesses explicit.
- It helps explain why fast path exists and where it is appropriate.

Estimated effort:
- implementation: `0.5 day`
- compute cost on Mac M2: low
- GPU required: no

### P0: Fast-Path Boundary and Benefit Report
Document and measure:
- which queries are eligible for fast path
- why they are eligible
- what fallback path is used on miss
- fast-path share of the benchmark/regression set
- latency delta between fast path and slow path

Deliverables:
- `docs/fast-path-boundary.md`
- metrics included in unified evaluation report

Why this matters:
- It reframes fast path as a controlled latency optimization layer, not a pile of one-off rules.

Estimated effort:
- implementation and docs: `0.5 day`
- compute cost on Mac M2: negligible
- GPU required: no

## P1: Productization and Stability

### P1: Structured Logging and Traceability
Add structured logs for each request:
- `query_id`
- selected path: `fast_path` / `slow_path`
- fallback reason
- retrieval candidate counts
- timings
- generator mode

Deliverables:
- logging helper in `src/core/`
- optional request trace block in API responses

Why this matters:
- It improves debuggability and makes expert review easier.
- It reduces ambiguity when performance regresses.

Estimated effort:
- implementation: `0.5 to 1 day`
- compute cost on Mac M2: negligible
- GPU required: no

### P1: Index Version Metadata
Persist index metadata with:
- corpus source
- chunk count
- chunking strategy
- encoder backend
- encoder model
- FAISS presence
- build timestamp
- build command hash or config hash

Deliverables:
- extend existing metadata in `artifacts/indexes/`
- surface index version in UI/API health endpoints

Why this matters:
- It makes the system more reproducible.
- It prevents confusion when comparing results across index rebuilds.

Estimated effort:
- implementation: `0.5 day`
- compute cost on Mac M2: negligible
- GPU required: no

### P1: Error Handling and Fallback Policy
Standardize failure behavior for:
- missing index files
- LLM provider timeouts
- unsupported question types
- insufficient evidence

Deliverables:
- explicit fallback reasons in pipeline output
- documented error policy in `README.md` or a new ops note

Why this matters:
- It makes the project feel less like a script bundle and more like a system.

Estimated effort:
- implementation: `0.5 to 1 day`
- compute cost on Mac M2: negligible
- GPU required: no

## P2: Optional Model Upgrades

### P2: Learned Reranker Exploration
Candidate options:
- local cross-encoder reranker
- hosted rerank API

Recommendation:
- keep current heuristic rerank as the default stable path
- treat learned rerank as a separate experiment, not a mandatory upgrade

Why this is not P0:
- The current heuristic rerank is acceptable for prototype-to-product transition.
- Learned reranking increases complexity and inference cost.

Estimated effort:
- implementation and evaluation: `2 to 4 days`
- compute cost on Mac M2: medium to high
- GPU required: no, but strongly helpful if the reranker is large

## Mac M2-Only Resource Assessment

## Summary Judgment
For the next phase, a Mac M2-class machine is sufficient.

You do not need to rent a GPU for the high-priority work in this phase because:
- the main tasks are metrics, bucketing, logging, reporting, and productization
- the dense FAISS index has already been rebuilt with transformer embeddings
- the current bottleneck is not training, but measurement and system finishing

The only task that may become awkward on CPU-only hardware is experimenting with a learned reranker. Even that is optional.

## Expected Runtime on Mac M2

### Already Observed Local Cost
From current project history, the following CPU-only costs are already known:
- full transformer index rebuild on the train corpus: about `2.5 to 3 hours`
- warm slow-path query after startup: roughly `sub-second to a few seconds`, depending on query type
- first heavy startup and preload: still expensive, but already operational

These numbers imply:
- offline rebuilds are slow but manageable on M2
- daily development does not require repeating the full index build often

### Next-Phase Tasks on Mac M2

1. Unified evaluation report
- expected runtime per report: `minutes`, not hours
- because it reuses existing indexes and benchmark/regression cases

2. Problem-type bucketing
- no meaningful compute burden

3. Fast-path hit-rate measurement
- no meaningful compute burden

4. Structured logging and metadata improvements
- no meaningful compute burden

5. Learned reranker experiment, if attempted
- small-model local inference on M2 CPU is possible
- practical estimate for a small evaluation batch: `tens of minutes to a few hours`, depending on candidate count
- full-corpus online use may feel too slow without careful top-k control

## Recommendation on GPU Rental

### GPU Not Needed For
- unified metrics report
- fast-path boundary analysis
- bucketed evaluation
- logging and observability
- fallback policy cleanup
- index version metadata
- UI/API productization

### GPU Might Be Worth It Only If
- you decide to replace heuristic rerank with a learned local reranker
- you want to iterate repeatedly on larger embedding models than `all-MiniLM-L6-v2`
- you want much faster repeated full-corpus index rebuilds

### Practical Recommendation
Do not rent a GPU for the next phase.

Use the Mac M2 machine for:
- all P0 tasks
- all P1 tasks
- small-scale P2 experiments only if needed

Reconsider GPU rental only after the current productization gaps are closed and only if:
- rerank quality is still the main blocker
- or repeated dense index rebuilds become frequent enough to hurt iteration speed

## Recommended Execution Order

### Week 1 or Equivalent
1. Build unified evaluation report
2. Add problem-type buckets
3. Add fast-path hit-rate and latency share reporting

Expected cost on Mac M2:
- about `1.5 to 3 days` of engineering work
- compute overhead: light

### Week 2 or Equivalent
1. Add structured logs
2. Add index version metadata
3. Standardize fallback and error handling
4. Surface more system state in API and UI

Expected cost on Mac M2:
- about `1.5 to 3 days` of engineering work
- compute overhead: light

### Later, Only If Needed
1. Try a learned reranker on a small evaluation slice
2. Compare against heuristic rerank
3. Decide whether the quality gain justifies the added complexity

Expected cost on Mac M2:
- `2 to 4 days` engineering and testing
- compute overhead: moderate to high

## Definition of Done For This Phase
This phase should stop when the project can clearly show:

1. a single report with retrieval, grounding, and latency metrics
2. bucketed results by question type
3. a clear explanation of what fast path does and how much it helps
4. request-level observability and stable fallback behavior
5. reproducible index version information

At that point, the project will still be a prototype, but it will be a much stronger, more review-ready, product-style prototype.
