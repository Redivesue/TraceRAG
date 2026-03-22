# Ops Note

## Purpose
This note describes the runtime behavior boundaries of the current Perplexity Lite system so that reviewers can understand how the application behaves under normal, degraded, and fallback conditions.

## Request Paths
The system has two main request paths:

1. `fast_path`
2. `slow_path`

### Fast Path
Fast path is a narrow optimization layer for simple structured entity questions such as:
- nationality
- profession
- older / younger comparisons
- same nationality / same profession / same birthplace comparisons

It should be understood as a latency optimization layer, not the main reasoning engine.

### Slow Path
Slow path is the default grounded QA route:

`query normalization -> BM25 + FAISS retrieval -> RRF fusion -> title dedup -> rerank -> grounded generation`

This path is used for:
- concept questions
- experiment questions
- open-ended grounded explanations
- any query that does not satisfy fast-path conditions

## Fallback Policy
The system follows this fallback order:

1. `fast_path` if a supported structured pattern matches and evidence is sufficient
2. hosted grounded generation if LLM configuration is available
3. offline grounded fallback if hosted generation is unavailable or fails
4. conservative answer if evidence is insufficient

Current visible path reasons include:
- `fast_path_match`
- `hosted_llm_success`
- `llm_not_configured_offline_fallback`
- `hosted_llm_error_offline_fallback`
- `no_evidence`

## Runtime Metadata
The runtime exposes:
- selected LLM provider and model
- dense encoder backend and model
- default chunk corpus path
- index manifest information

The API `/health` endpoint returns this metadata. The Streamlit UI shows it in `Runtime Status`.

## Request Traceability
Each request now has:
- `request_id`
- `selected_path`
- `path_reason`
- retrieval candidate count
- rerank candidate count
- stage timings

This information is exposed through:
- API search responses
- structured stdout logs
- the Streamlit `Request Trace` panel

## Failure Expectations
The system is expected to fail conservatively:
- if evidence is weak, the answer may remain incomplete
- if hosted LLM fails, the answer should fall back to offline grounded generation
- if fast path does not confidently match, the system should use slow path instead of forcing a rule answer

## Current Operational Limits
- first startup can still be heavy because retrievers and the transformer query encoder are preloaded
- slow-path quality is better than earlier iterations, but concept/experiment questions may still contain tail noise beyond top-1
- current evaluation reports are strongest as warm-runtime indicators; they are not yet a full cold-start SLA benchmark

## Reviewer Guidance
When reviewing the system, prefer asking:
- whether the chosen path was appropriate
- whether fallback behavior remained grounded
- whether index metadata matches the active runtime configuration
- whether fast path improves latency without becoming the primary answer engine
