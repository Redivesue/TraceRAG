# Perplexity Lite Development Plan

## Project Summary
This project will implement **Perplexity Lite: a retrieval-augmented QA system with reranking and citation grounding**. The goal is not to build a general chatbot, but a compact search-style AI system that can retrieve relevant evidence, reorder candidate passages, generate grounded answers, and show traceable citations.

Core value:

- Improve retrieval quality with hybrid search
- Improve ranking precision with semantic reranking
- Reduce hallucination through evidence-constrained generation
- Make answers explainable with chunk-level citations

## MVP Scope
The first version should support one end-to-end flow:

`User query -> query normalization -> retrieval -> rerank -> answer generation -> citation output`

In scope:

- HotpotQA small subset as the initial dataset
- BM25 retrieval over chunked passages
- Dense vector retrieval with sentence-transformers + FAISS
- Hybrid candidate fusion and deduplication
- Reranking over top retrieved passages
- LLM answer generation using only top-k evidence
- Citation display for each grounded claim
- Basic evaluation for retrieval and grounding

Out of scope for this phase:

- Multi-turn memory
- Web crawling
- User accounts
- Fine-tuning
- Complex agent workflows

## Dataset Decision
Current working decision:

- Primary build dataset: HotpotQA train plus dev_distractor
- Optional later extension: Natural Questions after access path is confirmed

Rationale:

- HotpotQA is the best fit for citation-grounded QA
- Natural Questions is valuable but currently less convenient for fast MVP execution

## Recommended Stack
Backend:

- Python
- FastAPI
- `rank-bm25`
- `sentence-transformers`
- `faiss`

Frontend:

- Streamlit for fastest demo delivery

Generation and rerank:

- Start with hosted LLM and hosted rerank API for speed
- Keep interfaces modular so a local cross-encoder reranker can replace the API later

## Suggested Repository Layout
Create the project with this structure:

- `src/api/` FastAPI routes
- `src/retrieval/` BM25, dense retrieval, hybrid fusion
- `src/rerank/` reranker client or local model wrapper
- `src/generation/` prompt builder and answer synthesis
- `src/data/` loaders, chunking, indexing scripts
- `src/core/` config, schemas, shared utilities
- `app/` Streamlit UI
- `tests/` unit and pipeline tests
- `artifacts/` generated indexes and cached subsets
- `docs/` plans, architecture notes, evaluation summaries

The initial scaffold should keep interfaces stable across modules:

- `SearchPipeline` is the single orchestration entrypoint
- `Retriever`, `Reranker`, and `AnswerGenerator` are swappable components
- API and UI should depend on pipeline output, not internal retriever details

## Module Call Graph
Planned runtime call path:

1. `src/api/main.py` receives `SearchRequest`
2. `src/pipeline/search_pipeline.py` runs the request
3. `src/retrieval/hybrid.py` calls `bm25.py` and `dense.py`
4. `src/rerank/hosted.py` reranks fused candidates
5. `src/generation/llm_generator.py` builds a citation-constrained answer and selects either a hosted LLM provider or the offline fallback
6. `SearchResponse` returns answer, citations, retrieved list, and reranked list

This separation is important because retrieval, rerank, and generation will likely change independently during iteration.

## Interface Contracts
Stabilize these interfaces before implementation details:

- `Retriever.retrieve(query, top_k) -> list[RetrievalCandidate]`
- `Reranker.rerank(query, candidates, top_k) -> list[RerankResult]`
- `AnswerGenerator.generate(query, evidence) -> GeneratedAnswer`
- `SearchPipeline.run(request) -> SearchResponse`

Shared schemas should live in `src/core/schemas.py` so every layer uses the same request and response contracts.

## Two-Day Execution Plan
### Day 1: Retrieval Pipeline
- Prepare a small HotpotQA subset and convert contexts into chunked passages with source IDs
- Build BM25 and FAISS indexes
- Implement top-k retrieval for both sparse and dense search
- Merge and deduplicate candidates into a unified retrieval list
- Add a rerank stage over the top 30 to 40 candidates
- Return top 5 evidence chunks for answer generation
- Finish interface-first scaffold so each module can be implemented independently

### Day 2: Grounded QA and Evaluation
- Add prompt constraints: answer only from evidence, cite sources, say unknown when unsupported
- Generate answers with chunk references like `[1]`, `[2]`
- Build a simple UI showing answer, citations, retrieved passages, and reranked passages
- Evaluate Recall@5, Recall@10, and MRR on a small validation subset
- Add lightweight groundedness checks such as citation hit rate
- Write a short results summary for resume and interview use

## Immediate Next Build Steps
After the scaffold is in place, implement in this order:

1. Data loader and chunker for HotpotQA subset
2. BM25 retriever over chunk corpus
3. Dense retriever with sentence-transformers plus FAISS
4. Hybrid fusion strategy with deduplication and score tracking
5. Hosted reranker adapter
6. LLM answer generator with strict citation prompt
7. FastAPI endpoint and Streamlit integration
8. Evaluation script for Recall@K, MRR, and citation hit rate

## Milestones and Deliverables
Milestone 1:
End-to-end retrieval plus rerank works from a single API endpoint.

Milestone 2:
The UI displays grounded answers and evidence snippets with stable citation IDs.

Milestone 3:
A small evaluation script reports retrieval and grounding metrics.

Expected deliverables:

- Runnable backend service
- Demo UI
- Sample indexed HotpotQA subset
- Evaluation script and metric output
- Architecture and resume-ready project summary

## Acceptance Criteria
The MVP is complete when:

- A user can ask a question and receive an answer with citations
- The system shows which chunks were retrieved and reranked
- The answer is restricted to retrieved evidence
- Retrieval metrics and at least one grounding metric are reported
- The codebase is modular enough to swap rerank and generation providers later
