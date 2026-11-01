# Validation status — 2026-11-01

## Exercised locally

Linux x86_64, Python 3.12, CPU inference. The core suite passes with no semantic dependencies/model required. A separate integration test loaded real E5-small with Python socket connections and DNS blocked, checked 384-dimensional output, indexed two contrasting articles, retrieved the expected article, and rejected over-limit input.

End-to-end commands exercised: model download, model-tokenized preparation, FP32 indexing, BM25/dense/hybrid search and evaluation, full local article reading, report comparison, INT8 AVX2 export, separate INT8 indexing/evaluation, and fresh-process FP32/INT8 offline checks. INT8 conversion was rerun after fixing exporter isolation, and source bundle fingerprint equality was checked before/after conversion.

Core tests cover Turkish dotted/dotless I, proper-name apostrophes, stemming/folding, negation retention, source offsets, complete prefixed token budgets, long-sentence splitting, overlap progress, duplicate IDs, altered prepared data, model/index mismatch, RRF ties, article deduplication, zero-match queries, known nDCG/MRR values, article-versus-evidence distinctions, question-family leakage, immutable output paths, paired report comparison, and fresh-process lexical offline checks.

Model revision: `intfloat/multilingual-e5-small` at `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Key tested dependency constraints are in `constraints-tested.txt`; complete measured package versions are included in `demo-results.json`.

## Synthetic smoke-test results

Twelve fictional articles, 21 model-tokenized passages, 20 dev queries (17 answerable, 3 unanswerable). These were AI-authored implementation fixtures, not independently labeled. No tuning against the six illustrative test questions was performed. Do not interpret the following as general Turkish, survival, or medical accuracy.

| Method | Article Success@1 | Article Success@3 | Warm P95, ms |
|---|---:|---:|---:|
| E5 FP32 | 15/17 | 15/17 | 26.57 |
| BM25 stemmed + folded | 14/17 | 16/17 | 0.37 |
| Hybrid RRF | 15/17 | 16/17 | 28.86 |
| E5 INT8 AVX2 | 14/17 | 15/17 | 9.21 |

These are single-run timings on this environment, not a controlled performance claim for your Mac. Default dense/hybrid configurations have no confidence threshold and accepted all three unanswerable questions. BM25 accepted two of three. This deliberately exposes the no-answer problem instead of hiding it in an aggregate accuracy score. Threshold calibration requires a larger reviewed dev set.

The final clean INT8 ONNX model file was ~118.3 MB; the complete exported bundle was ~136 MB with tokenizer/configuration. This excludes installed runtime libraries, articles, index and RAM. The documented `arm64` configuration is intended for an Apple Silicon CPU but was not tested on a Mac in this environment.

See `demo-results.json` for summaries, metrics, bootstrap intervals and environment metadata. Reproduce full per-query predictions locally with the README commands; large model/index artifacts are intentionally absent from GitHub.

## Known upstream warning

Transformers 4.57.6 emits a Mistral-regex warning when loading the locally re-saved E5 configuration. Inspection of its version-based detection showed that it classifies this non-Mistral local bundle as a candidate for that warning. This project does not apply a Mistral tokenizer rewrite to E5. Real-tokenizer/real-model integration passed; the warning remains visible rather than silently changing tokenization. Future dependency updates should be checked against the same integration tests and your held-out Turkish evaluations.

## Remaining work

Human-labeled target-domain evaluation; native Turkish question collection; confidence calibration; actual Mac/phone bundle, latency and memory measurements; content review; and any later reranker/fine-tuning implementation. CI is configured for model-free tests; it is not evidence that the real-model integration ran in GitHub Actions. This PR adds a research component, not a public emergency release.
