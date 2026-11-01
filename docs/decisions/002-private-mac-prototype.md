# Decision 002: Private Mac prototype

**Status: accepted for private testing. Not approved for real emergencies or public release.**

## Why
We need to test whether a Turkish earthquake assistant can find the right guidance and show useful answers with no internet. Decision 001 showed that Kumru-2B isn't safe for rewriting instructions. This prototype still lets an LLM answer every question, with the source text visible, so we can measure how other models fail. That's acceptable only in private testing.

## Decisions
- **Repo:** build in `survival-tr`.
- **Form:** local web app on a Mac. Phone and publishing decisions come later.
- **Scope:** earthquakes only, in Turkish only.
- **Testers:** the owner and people who know it is a test, using invented scenarios.
- **Models:** phone-sized models only.
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
Search comparison: not run yet.
