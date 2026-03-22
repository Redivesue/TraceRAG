# Perplexity Lite System Architecture

## End-to-End Flow
The MVP follows one synchronous pipeline:

`query -> normalize -> retrieve -> fuse -> rerank -> generate -> cite -> respond`

Execution order:

1. API or Streamlit receives a user query
2. `SearchPipeline` normalizes the query and requests candidate passages
3. `HybridRetriever` calls sparse and dense retrievers in parallel-friendly order
4. Fused candidates are deduplicated and sent to the reranker
5. Top reranked chunks are passed into the generator prompt
6. The generator returns an answer with citation IDs
7. The response includes answer text, citations, retrieved chunks, and reranked chunks

## Repository Skeleton
Recommended code layout:

```text
src/
  api/
    main.py
  core/
    config.py
    schemas.py
  data/
    loaders.py
    chunking.py
    indexing.py
  generation/
    base.py
    llm_generator.py
    openai_compatible.py
  pipeline/
    search_pipeline.py
  rerank/
    base.py
    hosted.py
  retrieval/
    base.py
    bm25.py
    dense.py
    hybrid.py
app/
  streamlit_app.py
tests/
docs/
artifacts/
```

## Module Responsibilities
### `src/core/schemas.py`
Defines shared data contracts. Main objects:

- `SearchRequest`: query, top_k, debug flag
- `Chunk`: chunk ID, source title, text, metadata
- `RetrievalCandidate`: chunk plus retrieval score and source method
- `RerankResult`: chunk plus rerank score and final rank
- `SearchResponse`: answer, citations, retrieved candidates, reranked candidates

### `src/retrieval/`
- `base.py`: retriever interface
- `bm25.py`: sparse retrieval over tokenized chunks
- `dense.py`: embedding retrieval with FAISS
- `hybrid.py`: merges sparse and dense candidates, applies reciprocal rank fusion or weighted score fusion

### `src/rerank/`
- `base.py`: reranker interface
- `hosted.py`: hosted rerank API client

Input:
query + candidate chunks

Output:
same chunks reordered with rerank scores

### `src/generation/`
- `base.py`: generator interface
- `llm_generator.py`: prompt builder and provider selection
- `openai_compatible.py`: hosted chat completion client for OpenAI-compatible endpoints

Input:
query + top reranked chunks

Output:
grounded answer with explicit citation markers like `[1]`

### `src/pipeline/search_pipeline.py`
Orchestrates the complete request. This is the only layer that should know all downstream modules.

## Primary Interfaces
Use these stable method contracts:

```python
Retriever.retrieve(query: str, top_k: int) -> list[RetrievalCandidate]
Reranker.rerank(query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RerankResult]
AnswerGenerator.generate(query: str, evidence: list[RerankResult]) -> GeneratedAnswer
SearchPipeline.run(request: SearchRequest) -> SearchResponse
```

Rules:

- Retrievers do not call rerankers or generators directly
- Rerankers only reorder candidates and must not generate text
- Generators only consume reranked evidence, never raw corpus access
- API and UI call `SearchPipeline`, not lower-level modules

## API Contract
Initial endpoint:

- `POST /search`

Request body:

```json
{
  "query": "Which scientist proposed relativity?",
  "top_k": 5,
  "debug": true
}
```

Response shape:

```json
{
  "answer": "Albert Einstein proposed the theory of relativity [1].",
  "citations": [{"label": 1, "chunk_id": "doc12_chunk3", "title": "Albert Einstein"}],
  "retrieved": [],
  "reranked": []
}
```

## Data and Indexing
HotpotQA subset processing should produce:

- `artifacts/hotpotqa/dev_subset.jsonl`
- `artifacts/indexes/bm25/`
- `artifacts/indexes/faiss/`
- `artifacts/chunks/chunks.jsonl`

Each chunk should keep:

- `chunk_id`
- `doc_id`
- `title`
- `text`
- `supporting_fact_hint` when available

## Design Constraints
- Keep interfaces provider-agnostic so hosted services can later be swapped for local models
- Make hosted LLM access optional so the system still runs in offline mode for evaluation and local debugging
- Keep scoring fields from each stage for later evaluation and debugging
- Return debug payloads in the response so the UI can show retrieval before and after reranking
- Treat citation IDs as stable labels mapped to concrete chunk IDs
