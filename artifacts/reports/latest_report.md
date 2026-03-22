# Unified Evaluation Report

- Generated at: `2026-03-22T09:44:28.971794+00:00`
- Eval split: `artifacts/raw/hotpotqa/hotpot_dev_distractor_v1.json`
- Eval limit: `5`
- Benchmark rounds: `1`

## Retrieval Metrics

- Evaluated examples: `5`
- Recall@5: `1.0000`
- Recall@10: `1.0000`
- MRR: `0.7333`
- Citation hit rate: `1.0000`
- Citation precision: `0.7000`
- Answer grounded rate: `1.0000`

## Regression Summary

- Case count: `7`
- Pass rate: `1.0000`
- Fast-path hit rate: `0.5714`
- Average total latency: `0.0218s`

## Fast Path Benefit

- Fast-path cases: `4`
- Slow-path cases: `3`
- Fast-path avg latency: `0.0002s`
- Slow-path avg latency: `0.0505s`
- Slow/Fast speedup ratio: `224.30x`

## Bucket Summary

### comparison_age

- Count: `1`
- Pass rate: `1.0000`
- Grounded rate: `1.0000`
- Fast-path hit rate: `1.0000`
- Avg total latency: `0.0001s`

### comparison_nationality

- Count: `1`
- Pass rate: `1.0000`
- Grounded rate: `1.0000`
- Fast-path hit rate: `1.0000`
- Avg total latency: `0.0004s`

### concept_experiment

- Count: `2`
- Pass rate: `1.0000`
- Grounded rate: `1.0000`
- Fast-path hit rate: `0.0000`
- Avg total latency: `0.0754s`

### entity_nationality

- Count: `1`
- Pass rate: `1.0000`
- Grounded rate: `1.0000`
- Fast-path hit rate: `1.0000`
- Avg total latency: `0.0001s`

### entity_profession

- Count: `1`
- Pass rate: `1.0000`
- Grounded rate: `1.0000`
- Fast-path hit rate: `1.0000`
- Avg total latency: `0.0003s`

### guardrail_birthplace

- Count: `1`
- Pass rate: `1.0000`
- Grounded rate: `1.0000`
- Fast-path hit rate: `0.0000`
- Avg total latency: `0.0005s`

## Latency Summary

- retrieval_seconds: mean=`0.0000s`, p50=`0.0000s`, p95=`0.0000s`, max=`0.0000s`
- rerank_seconds: mean=`0.0001s`, p50=`0.0000s`, p95=`0.0004s`, max=`0.0004s`
- generation_seconds: mean=`0.0001s`, p50=`0.0000s`, p95=`0.0003s`, max=`0.0003s`
- total_seconds: mean=`0.0003s`, p50=`0.0001s`, p95=`0.0007s`, max=`0.0007s`

## Generator Modes

- fast_path: count=`4`, mean=`0.0001s`, p50=`0.0001s`, p95=`0.0001s`
- offline: count=`3`, mean=`0.0005s`, p50=`0.0007s`, p95=`0.0007s`

## Notes

- `answer_grounded_rate` is currently approximated as the share of evaluation examples whose used citations overlap with HotpotQA supporting-fact titles.
- `citation_precision` is averaged at the example level using cited-title overlap against supporting-fact titles.
- Regression buckets are lightweight product-facing slices rather than a benchmark-standard taxonomy.
