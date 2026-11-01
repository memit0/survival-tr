# Turkish Offline Search

A standalone experimental Turkish article search engine. It retrieves original excerpts and full articles; it never generates an answer. Everything after dependency/model installation runs locally. Python 3.10+; start on a laptop, measure before deploying to a phone.

**Status:** working research prototype, not an emergency product. The included collection describes fictional library/account services. Its 26 AI-authored questions (20 dev, 6 test) are transparent smoke-test fixtures, **not a human-validated accuracy benchmark**. The [Survival TR release requirements](docs/safety-policy.md#release-requirements) still apply to any future emergency use.

## Quick start on macOS / Linux

From the root of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-tested.txt -e '.[semantic,dev]'

# ONLINE once: download E5-small (~hundreds of MB, plus runtime dependencies).
tr-search download-model --output models/e5-small

# OFFLINE from here: clean, token-aware passages; exact vectors + BM25.
tr-search prepare --articles data/demo/articles.jsonl --output prepared/demo.json --model models/e5-small
tr-search index --prepared prepared/demo.json --output indexes/demo --model models/e5-small --stem --ascii-fold
tr-search search --index indexes/demo --model models/e5-small 'Parolamı hatırlamıyorum nasıl yenilerim?'
tr-search search --index indexes/demo --method hybrid --model models/e5-small 'kutuphane pazar acik mi'

# Read the original article entirely from SQLite.
tr-search article --index indexes/demo password-reset

# A fresh process blocks Python socket connections/DNS BEFORE loading the model.
tr-search offline-check --index indexes/demo --model models/e5-small 'Notlarımı internetsiz nasıl okuyabilirim?' --json
```

Also repeat with Wi-Fi disconnected. The Python socket guard is a diagnostic, **not an OS-level network sandbox** and cannot intercept networking from every native library. The local-only loaders and offline environment flags prevent intentional Hub calls. No API keys or servers required. Model download is the only deliberately network-enabled model command. On some proxied development machines, installing `socksio` may be necessary for downloads.

Commands refuse to overwrite existing model, index, prepared-data, or report destinations. Use a new path per experiment. Return code 2 indicates invalid input/configuration; search with no results returns 0 with `accepted: false` in JSON.

### Try the lightweight keyword baseline first

No model, Torch, or internet is needed after installing core dependencies:

```bash
python -m pip install -e '.[dev]'
tr-search prepare --lexical-only --articles data/demo/articles.jsonl --output prepared/lexical.json
tr-search index --lexical-only --prepared prepared/lexical.json --output indexes/lexical --stem --ascii-fold
tr-search search --index indexes/lexical --method bm25 'kütüphane üyeliği'
```

`--lexical-only` preparation counts regex word/punctuation units, **not E5 tokens**. For fair model comparisons use the same model-tokenized prepared file for dense and BM25 indexes. BM25 is implemented locally with positive IDF, `k1=1.5`, `b=0.75`, no stopword removal, optional Snowball Turkish stemming, optional Turkish ASCII folding. Exact-name apostrophe suffixes are removed in the lexical branch. Stemmed/folded text is never sent to E5.

## Evaluate and compare

Use the same index and questions to isolate the retrieval-method change:

```bash
tr-search evaluate --index indexes/demo --model models/e5-small --method dense --queries data/demo/queries.jsonl --output reports/dense
tr-search evaluate --index indexes/demo --method bm25 --queries data/demo/queries.jsonl --output reports/bm25
tr-search evaluate --index indexes/demo --model models/e5-small --method hybrid --queries data/demo/queries.jsonl --output reports/hybrid
tr-search compare reports/dense reports/hybrid
```

Each report contains:

- `report.md`: readable metrics and family-cluster bootstrap intervals.
- `summary.json`: configuration, package versions, fingerprints, warm latency and index bytes.
- `predictions.jsonl`: every question and ranked article/passage result.
- `failures.json`: first-article misses and false accepts on unanswerable questions.

Main metrics: distinct-article Success@1/@3, Recall@3 (grade 2), MRR@10, graded nDCG@10; evidence-containing excerpt Success@1 and candidate Success@20; false acceptance/rejection rates. Reported success means matching labels, not independent factual verification. Unjudged articles count as 0; pool and review candidates to make labels sufficiently complete. Queries with no evidence spans are excluded from evidence metrics, not counted as successes or failures.

Read [the evaluation guide](docs/evaluation.md) before interpreting scores or tuning on a final test set.

## Bring your own Turkish articles

Create UTF-8 JSONL: one JSON object per line. IDs must be unique. Preserve text exactly once you label evidence.

```json
{"article_id":"my-article","title":"Başlık","source":"Source URL or citation","version":"1","text":"# Bölüm\nMakalenin özgün metni..."}
```

The prototype accepts text/Markdown inside JSONL; it does not crawl websites, parse PDFs, or collect official emergency guidance. Clean extraction separately. Markdown headings set section context. Passages retain their parent article ID and exact character offsets into original text. Defaults are 240 model tokens including `passage:`, title, heading, and special tokens; overlap is one sentence/word unit. Long sentences split on whitespace. An unbreakable over-budget word or over-budget heading raises an error instead of truncating silently.

For a different experiment:

```bash
tr-search prepare --articles data/my-articles.jsonl --output prepared/my-320.json --budget 320 --overlap 1
tr-search index --prepared prepared/my-320.json --output indexes/my-320 --stem
```

## INT8 ONNX experiment

Install conversion dependencies while online, then export from the local FP32 bundle:

```bash
python -m pip install -c constraints-tested.txt -e '.[onnx]'
tr-search quantize --source models/e5-small --output models/e5-int8 --config arm64
```

Use `arm64` for an Apple Silicon CPU; use a suitable `avx2`/`avx512` configuration for an x86 CPU. This is CPU inference. E5 uses two Torch threads by default; override with `TR_SEARCH_THREADS=4` and record the setting when benchmarking. Speedups and accuracy depend on hardware; compare against FP32. The exporter prints relative ONNX filenames; use its exact output (some configurations use `quint8` rather than `qint8`). For an export reported as `onnx/model_qint8_arm64.onnx`, pass that complete relative path as `--onnx-file onnx/model_qint8_arm64.onnx`:

```bash
tr-search index --prepared prepared/demo.json --output indexes/demo-int8 --model models/e5-int8 --backend onnx --onnx-file onnx/model_qint8_arm64.onnx --stem --ascii-fold
tr-search evaluate --index indexes/demo-int8 --model models/e5-int8 --backend onnx --onnx-file onnx/model_qint8_arm64.onnx --queries data/demo/queries.jsonl --output reports/int8
tr-search compare reports/dense reports/int8
```

The quantized bundle omits FP32 weights; conversion runs in a temporary copy so the source bundle is unchanged. Quantized and FP32 encoders get separate passage vectors; an index rejects a different model/backend/file. Keep complete model directories available offline. Do not assume the model bundle size equals RAM or total installed package size.

## Architecture

The project has one job: find the original article text that best matches a Turkish question. No model writes or rewrites text, so what you read is always the source wording. [docs/architecture.md](docs/architecture.md) defines the terms used below.

### The big picture

The network is used only to install packages and download the model. Everything after that runs locally, from four kinds of files on disk.

```mermaid
flowchart LR
    subgraph online["Online, once"]
        HF["Hugging Face<br/>E5-small model"]
    end
    subgraph build["Offline: build"]
        A["data/*.jsonl<br/>your articles"]
        P["prepared/*.json<br/>validated passages"]
        I["indexes/*/<br/>SQLite + vectors + manifest"]
        A -->|tr-search prepare| P
        P -->|tr-search index| I
    end
    subgraph use["Offline: use"]
        S["search<br/>offline-check"]
        V["article<br/>full original text"]
        E["evaluate<br/>reports/*/"]
        C["compare<br/>two reports"]
        E --> C
    end
    HF -->|tr-search download-model| M["models/e5-small"]
    M -.->|tokenizer counts tokens| P
    M -.->|encodes passages| I
    M -.->|encodes questions| S
    M -.->|encodes questions| E
    I --> S
    I --> V
    I --> E
```

Dotted lines are skipped in keyword-only (`--lexical-only` / `--method bm25`) mode, which needs no model at all.

### Building an index

`prepare` splits articles without changing them, so every passage can be traced to exact character positions in the original. `index` then writes everything to a temporary folder and renames it into place only if all steps succeed, so a failed build never leaves a half-finished index.

```mermaid
flowchart TD
    A["articles.jsonl"] --> V{"Valid?<br/>unique IDs, non-empty<br/>title and text"}
    V -->|no| X1["Stop with an error, exit code 2"]
    V -->|yes| H["Split on Markdown headings<br/>heading becomes section context"]
    H --> U["Split sections into sentences<br/>long sentences split on spaces"]
    U --> T{"Fits the budget?<br/>240 tokens including prefix,<br/>title and heading"}
    T -->|"a single word is too long"| X2["Stop with an error,<br/>never truncate silently"]
    T -->|yes| C["Group units into passages<br/>one unit of overlap<br/>keep start and end offsets"]
    C --> PJ["prepared/*.json<br/>with SHA-256 fingerprint"]
    PJ --> F{"Fingerprint still matches?"}
    F -->|no| X3["Stop: rerun prepare"]
    F -->|yes| TMP["Temporary folder"]
    TMP --> SQL["corpus.sqlite<br/>full articles and passages"]
    TMP --> ENC["E5 encoder<br/>input: passage: title, heading, text"]
    ENC --> NPY["vectors.npy<br/>one 384-number row per passage"]
    TMP --> MAN["manifest.json<br/>settings and model fingerprint"]
    SQL --> R["Rename to the final folder<br/>existing folders are refused"]
    NPY --> R
    MAN --> R
```

### Answering a search

Dense search matches meaning, BM25 matches words, and hybrid combines their rankings. Results then show the top articles with up to two passages each, because an article's best-scoring passage can be the wrong section.

```mermaid
flowchart TD
    L["Load the index<br/>read-only SQLite, memory-mapped vectors<br/>build BM25 word lists in RAM"] --> CHECK{"Model and backend<br/>match the manifest?"}
    CHECK -->|no| X["Stop: rebuild the index"]
    CHECK -->|yes| Q["Turkish question"]
    Q --> EMPTY{"Contains any words?"}
    EMPTY -->|no| NONE["accepted: false, no results"]
    EMPTY -->|yes| M{"--method"}

    M -->|dense or hybrid| DE["E5 encoder<br/>input: query: question<br/>unchanged Turkish text"]
    DE --> DS["Dot product with every passage vector<br/>exact, no approximate index"]
    DS --> DR["Dense ranking"]

    M -->|bm25 or hybrid| AN["Turkish text analysis<br/>dotted and dotless I, lowercase,<br/>remove apostrophe suffixes,<br/>optional stemming and ASCII folding"]
    AN --> BS["BM25 score<br/>k1 = 1.5, b = 0.75"]
    BS --> BR["Keyword ranking<br/>positive scores only"]

    DR -->|dense only| RANK["Final passage ranking"]
    BR -->|bm25 only| RANK
    DR -->|hybrid: top 50| RRF["Rank fusion, RRF<br/>sum of 1 / (60 + rank)"]
    BR -->|hybrid: top 50| RRF
    RRF --> RANK

    RANK --> CUT["Optional --min-dense-score filter<br/>off by default"]
    CUT --> DEDUP["Keep up to 2 passages per article<br/>from the top 3 articles"]
    DEDUP --> OUT["Original excerpt, article ID, source, score<br/>accepted: true if anything is left"]
```

For example, `kutuphane pazar acik mi` is typed without Turkish letters. In the demo evaluation, dense ranks the correct article 7th and BM25 ranks it 1st, so hybrid also ranks it 1st. The reverse happens too: `Odayı kaç saat ayırtabilirim?` is 3rd in BM25 and 1st in dense and hybrid.

`accepted: true` only means a result survived. It does not mean the answer is correct. An index built with `--examples` can also refuse: when the question contains a listed `no_answer_terms` word or sits closest to an `out_of_scope` example question, search returns `accepted: false` with a `no_answer` reason. This catches listed topics only; other unanswerable questions still get results.

### Measuring quality

`evaluate` scores a set of labeled questions and saves every prediction. `compare` then checks two reports question by question, and refuses if they used different questions or articles.

```mermaid
flowchart LR
    Q["queries.jsonl<br/>question, split, family,<br/>labels, evidence offsets"] --> V{"Valid labels?<br/>no question family<br/>in two splits"}
    V -->|yes| SP["Pick one split<br/>dev by default"]
    SP --> RUN["Search each question<br/>with the chosen method"]
    RUN --> SC["Score it<br/>article found? excerpt contains evidence?<br/>unanswerable question accepted?"]
    SC --> BOOT["Averages + 95% intervals<br/>bootstrap by question family"]
    BOOT --> OUT["report.md<br/>summary.json<br/>predictions.jsonl<br/>failures.json"]
    OUT --> CMP["tr-search compare<br/>right minus left"]
```

### Where the code lives

| File | Responsibility |
|---|---|
| `src/turkish_search/cli.py` | The `tr-search` commands, and the socket block used by `offline-check` |
| `src/turkish_search/data.py` | Reading and validating articles, passage splitting, fingerprints |
| `src/turkish_search/text.py` | Turkish text analysis, used for BM25 only |
| `src/turkish_search/encoder.py` | Loading local models, downloading, INT8 export, `query:`/`passage:` prefixes |
| `src/turkish_search/search.py` | Index writing, BM25, rank fusion, the `Searcher` |
| `src/turkish_search/evaluation.py` | Label validation, metrics, reports, comparison |
| `tests/` | `test_core.py` needs no model; `test_semantic.py` runs when `TR_SEARCH_MODEL` is set |
| `data/demo/` | 12 fictional articles and 26 AI-written questions |
| `docs/` | Glossary, evaluation guide, validation status, safety policy, decision records |

### Design details and limitations

- E5-small: 384-dimensional normalized vectors, explicit `query:`/`passage:` roles.
- SQLite contains full articles and passages; a NumPy `.npy` matrix contains exact-search FP32 vectors. No pickle deserialization or hosted vector database.
- Dense search compares against every passage. Hybrid combines the top 50 dense/BM25 passage lists with RRF (`k=60`), then selects each distinct article's highest-ranked passage. The candidate cutoff affects hybrid recall.
- `--min-dense-score` optionally filters dense/hybrid candidates by raw cosine score. No default cutoff: scores are not probabilities. BM25 accepts any positive lexical match. Dense without a cutoff normally accepts something even for an unanswerable question.
- No reranker, ANN index, training pipeline, mobile UI, web server, or production confidence model in v0.1. Add complexity only after failure analysis.
- `summary.json` includes index size, not complete installed-runtime size or peak RAM. Searcher load time excludes encoder initialization. Measure complete fresh-process startup and peak memory separately (see docs).

See [architecture and glossary](docs/architecture.md), [evaluation and optimization](docs/evaluation.md), and [validation status](docs/validation.md).

## Project history

This repository began as Survival TR, an offline emergency assistant. The earlier Kumru writer experiment [failed](docs/decisions/001-phase1-kumru-writer.md), so this project retrieves original text instead of generating it. The removed experiment code remains in git history at commit `f0e4dab`.

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Core tests require no model download. Semantic integration is separately exercised when a local E5 bundle exists; see `tests/test_semantic.py`. CI runs deterministic core tests and the BM25 CLI/eval path without downloading model weights.

## Research and documentation

- [E5 technical report](https://arxiv.org/abs/2402.05672) and [E5-small model card](https://huggingface.co/intfloat/multilingual-e5-small)
- [TR-MTEB (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.471/)
- [TR-TEB (LREC 2026)](https://aclanthology.org/2026.lrec-1.862/)
- [TurkColBERT](https://arxiv.org/abs/2511.16528)
- [Sentence Transformers efficiency / quantization](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)

MIT for this prototype. External article and model licenses remain their owners' responsibility. Demo articles and labels are synthetic and included under this project's MIT license.
