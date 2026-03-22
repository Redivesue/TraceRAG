# Dataset Sources and Download Status

## Recommended Public Datasets
### HotpotQA
- Role: primary dataset family for the MVP
- Use split roles:
  - `hotpot_train_v1.1.json` for corpus building and indexing
  - `hotpot_dev_distractor_v1.json` for retrieval and grounding evaluation
- Why it fits: question structure is clean and supporting facts make evidence display easier
- Official project: `https://github.com/hotpotqa/hotpot`
- Planned local path: `artifacts/raw/hotpotqa/`
- Current status: both official target files are being downloaded from the documented URLs; the official host is reachable but slow and unstable in this environment
  - `hotpot_dev_distractor_v1.json` is now fully downloaded and JSON-validated
  - `hotpot_train_v1.1.json` is still downloading incrementally from the official source

### Natural Questions
- Role: secondary dataset for search-like real user questions
- Why it fits: closer to realistic query distribution than benchmark-style QA alone
- Official project: `https://github.com/google-research-datasets/natural-questions`
- Planned local path: `artifacts/raw/natural_questions/`
- Current status: direct anonymous download attempt returned `403`; likely requires a different official access path or authenticated workflow

## Working Recommendation
Use `HotpotQA train` as the corpus source and `HotpotQA dev_distractor` as the evaluation source. Treat `Natural Questions` as optional until access is resolved.

## Files Downloaded So Far
- `artifacts/raw/hotpotqa/hotpot_dev_distractor_v1.json` (complete and validated)
- `artifacts/raw/hotpotqa/hotpot_train_v1.1.json` (partial, official source, still downloading)

## Next Data Tasks
1. Finish the official HotpotQA train download for corpus building.
2. Decide whether to skip Natural Questions in the first milestone.
3. Build a HotpotQA subset exporter into `artifacts/chunks/` for indexing.
