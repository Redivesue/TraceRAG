# Fast Path Boundary

## Purpose
The fast path is a latency and precision optimization layer for simple, structured, entity-centric questions.

It is not the main reasoning engine of the system. The main search QA path remains:

`normalize -> BM25 + FAISS -> RRF -> dedup -> rerank -> grounded generation`

The fast path exists to avoid paying the full slow-path cost when the question can be answered reliably from a very small, high-precision evidence set.

## Design Position
The fast path should be understood as:
- a controlled optimization layer
- narrow in scope
- high precision
- easy to fall back from

It should not become:
- the default answer engine for arbitrary questions
- a substitute for the retrieval + rerank pipeline
- an ever-growing bag of brittle rules

## Current Eligibility
The current implementation in [llm_generator.py](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/src/generation/llm_generator.py) supports these patterns when evidence is sufficient:

### Single-Entity Attribute Questions
- nationality
  - `What nationality is Scott Derrickson?`
- profession
  - `What is Ed Wood's profession?`
- birthplace
  - `Where was X born?`
  - `What city was X born in?`
  - `Born in which city/town`

### Comparison Questions
- same nationality
  - `Were A and B of the same nationality?`
- same profession / same job
  - `Did A and B have the same profession?`
- same birthplace / same city / same town
  - `Did A and B have the same birthplace?`
- older / younger
  - `Who is older, A or B?`

## What Fast Path Requires
Fast path is only used when all of the following are true:

1. the question matches a supported pattern
2. named entities can be extracted from the query
3. the retrieved evidence contains the required attribute explicitly
4. the answer can be formed without free-form synthesis beyond the supported templates

If any of these fail, the system should fall back to the normal grounded generation path.

## Why These Types Were Chosen
These question classes were selected because they have:
- relatively stable surface forms
- explicit evidence patterns
- low ambiguity when evidence is present
- strong latency sensitivity in a demo or product UI

They are good candidates for fast path because the answer often reduces to:
- extract one attribute
- compare two extracted attributes
- format a short grounded answer with citations

## Current Boundary
Fast path should be considered in-bounds for:
- short entity questions
- two-entity attribute comparison questions
- questions whose answer can be directly mapped from evidence text

Fast path should be considered out-of-bounds for:
- concept explanation
- experiment purpose/mechanism questions
- open-ended “what is X about” questions
- multi-hop reasoning with implicit evidence
- questions where evidence is incomplete or indirect

Examples that should stay on slow path:
- `What did the Pound–Rebka experiment test?`
- `What is the purpose of the Nucifer experiment?`
- `What is general relativity about?`

## Fallback Strategy
When fast path does not apply, the system falls back in this order:

1. hosted grounded generation if the LLM provider is configured
2. offline grounded fallback if hosted generation is unavailable

The fallback must preserve:
- evidence grounding
- citation filtering
- conservative behavior when evidence is insufficient

## How To Measure It
The unified report at [latest_report.md](/Users/hanwu/Documents/量化或大模型项目/LLM-Project1/artifacts/reports/latest_report.md) now includes:
- fast-path hit rate
- bucket-level pass rate
- bucket-level latency
- fast-vs-slow average latency comparison

These metrics are intended to answer four review questions:
- how often fast path triggers
- on which question types it triggers
- whether it preserves correctness
- how much latency it saves

## Practical Interpretation
Fast path is valuable when:
- it covers a meaningful share of common entity questions
- it remains significantly faster than slow path
- it preserves or improves precision
- it degrades safely to slow path on misses

Fast path becomes risky when:
- too many new question types are added ad hoc
- trigger rules become opaque
- behavior diverges too far from the main grounded QA path

## Current Recommendation
Keep fast path, but treat it as a narrow optimization layer.

In future iterations:
- continue to measure hit rate and latency benefit
- avoid expanding it to broad open-ended question types
- prefer better retrieval and rerank over endlessly extending rule coverage
