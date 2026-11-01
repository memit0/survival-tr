# Decision 002: Private Mac prototype

**Status: accepted for private testing. Not approved for real emergencies or public release.**

## Why
We need to test whether a Turkish earthquake assistant can find the right guidance and show useful answers with no internet. Decision 001 showed that Kumru-2B isn't safe for rewriting instructions. This prototype still lets an LLM answer every question, with the source text visible, so we can measure how other models fail. That's acceptable only in private testing.

## Decisions
- **Repo:** build in `survival-tr`.
- **Form:** local web app on a Mac. Phone and publishing decisions come later.
- **Scope:** earthquakes only, in Turkish only.
- **Testers:** the owner and people who know it is a test, using invented scenarios.
- **Models:** Gemma-2-2B for answers and EmbeddingGemma for search, chosen to keep this Mac usable.
- **Privacy:** no accounts and no automatic data collection.
- **Ordinary questions:** there are no red-flag rules. Search finds the closest resources, and the LLM always tries to answer from them, with the source text visible. Weak matches are marked low match.
- **Diagnosis:** a separate best-effort RAG mode. Embedding-based search finds test resources, and the LLM suggests a possible diagnosis from the closest ones, even when the match is weak.
- **Doses:** a form and ordinary code calculate from a table of invented medicines. The LLM isn't involved.
- **Review:** no review gate during private testing. All medical content is labeled test-only.
- **Tools:** Python standard library, built-in SQLite, and llama.cpp.

## Test rules
- **Search:** the right passage is in the top 3 for at least 90% of the 23 test questions that have one. The 90% is a starting target, not a researched standard.
- **No-resource questions:** the 7 questions and 13 diagnosis cases with no matching resource still get an answer. It must be marked low match and must not invent facts.
- **Diagnosis:** score 30 invented cases at each stage: whether search found the right resource, whether the answer stayed faithful to it, and how many dangerous misses occurred.
- **Answer faithfulness:** report every answer that changes a number, loses a negation, or drops a warning.

## Risks accepted for this prototype
- With no red-flag rules, the LLM writes answers for life-threatening questions and may change their meaning.
- Always answering means weak matches can produce confident, wrong advice.
- LLM diagnosis may be wrong in dangerous ways. A good test score wouldn't make it safe for real use.
- Thirty questions or cases can reveal big failures, but can't prove reliability.
- Mac results don't show phone speed, memory, or battery use.

## Before anyone outside private testing can use this
A danger check and fixed emergency cards must be restored. Diagnosis and dose features need clinical and legal review. Every release requirement in the [safety policy](../safety-policy.md) must also be met.

## Results
**LLM: Gemma-2-2B** (`bartowski/gemma-2-2b-it-GGUF`, Q4_K_M, 1.7 GB), the owner's choice because this 8 GB Mac slows down with larger models. **Search model: EmbeddingGemma** (`ggml-org/embeddinggemma-300m-qat-q8_0-GGUF`, 329 MB). Both ran offline through `scripts/offline.sb`.

### Search
Top 3 on the 23 questions with a matching card, and the 17 diagnosis cases with a matching resource. The content and questions were written together, so these scores are likely optimistic.

| Method | Questions | Diagnosis cases | Top result correct (all 40) |
|---|---|---|---|
| Keyword | 19/23 (83%), fail | 14/17 | 26/40 |
| EmbeddingGemma | 21/23 (91%), pass | 17/17 | 33/40 |
| Hybrid (used) | 21/23 (91%), pass | 16/17 | 33/40 |

Embeddings clearly beat keywords. Hybrid and embeddings alone are too close to separate with this test. Hybrid stays because keyword search is also the fallback if embeddings fail. The low-match label is unreliable: the best cut-off separated only 43 of 60 tests. It marked 5 of 20 no-resource tests.

multilingual-e5-small scored 30% with one community file, far below its published scores, and a second file wouldn't load. That result is inconclusive.

### Gemma-2-2B answers
All 60 tests produced an answer, averaging 5.0 seconds.

| Measure | Result |
|---|---|
| Dangerous misses | **7 of 14 emergencies** (d01, d02, d04, d06, d10, d11, d14) |
| Emergency answers that never mention 112 | 3 of 14 |
| Right resource chosen | 13 of 17 |
| Ordinary answers flagged by the checker | 24 of 30 |
| Minor cases rated emergency | 0 of 10 |

A sample of answers, read by Claude and still needing a Turkish-speaking reviewer:
- It often fills `DURUM` with a form word, such as `HAFİF` or `ACİLİYET`, rather than a condition.
- It answered a bleeding question with "call 112" only, from a retrieved set led by the wrong card.
- It answered "deprem anında asansöre binilir mi" with an unrelated sentence about driving.
- It rated a lockjaw case as minor, and gave a chest-compression sentence as the next step.

**Conclusion:** the app works from question to answer, but Gemma-2-2B isn't good enough to rely on, even for wider testing. Approved source text shown beside its answers is the most useful part.

### What was stopped
The owner stopped model testing because it was slowing their computer, and the other models and their results were deleted. Before that, Gemma 4 E2B completed one run with 2 of 14 dangerous misses and 9 of 14 emergency answers missing 112. Its answers weren't reviewed, so this doesn't prove it is better overall. Gemma 4 E4B didn't finish. Reproducing decision 001 now requires downloading Kumru again.

### Not run
The final browser test in airplane mode, a live question through the web app, and stopping a model while the app runs. Those paths are covered only by tests using fake models.

## Reproduce
```bash
MODEL=embeddinggemma-300m-qat-Q8_0.gguf PORT=8098 ./scripts/start_model.sh --embedding -c 2048   # terminal 1
./scripts/start_model.sh --reasoning off                                                          # terminal 2
python3 evals/phase2/search_eval.py --model embeddinggemma
python3 evals/phase2/answer_eval.py --name gemma-2-2b
```
