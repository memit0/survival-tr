# Decision 002: Private Mac prototype

**Status: accepted for private testing. Not approved for real emergencies or public release.**

## Why
We need to test whether a Turkish earthquake assistant can find the right guidance and show useful answers with no internet. Decision 001 showed that Kumru-2B isn't safe for rewriting instructions, so approved text stays the default.

## Decisions
- **Repo:** build in `survival-tr`.
- **Form:** local web app on a Mac. Phone and publishing decisions come later.
- **Scope:** earthquakes only, in Turkish only.
- **Testers:** the owner and people who know it is a test, using invented scenarios.
- **Models:** phone-sized models only.
- **Privacy:** no accounts and no automatic data collection.
- **Ordinary questions:** fixed danger checks run first, then search shows an approved passage or says it doesn't know.
- **Diagnosis:** a separate mode where the LLM suggests a possible diagnosis. Any matching 112 card appears above it.
- **Doses:** a form and ordinary code calculate from a table of invented medicines. The LLM isn't involved.
- **Review:** no review gate during private testing. All medical content is labeled test-only.
- **Tools:** Python standard library, built-in SQLite, and llama.cpp.

## Test rules
- **Search:** the right passage is in the top 3 for at least 90% of test questions, and 100% of red-flag questions reach the correct card. The 90% is a starting target, not a researched standard.
- **Diagnosis:** score 30 invented cases, and report dangerous misses separately from accuracy.
- **Rewriting:** no approved output may change a number, lose a negation, or drop a warning.

## Risks accepted for this prototype
- LLM diagnosis may be wrong in dangerous ways. A good test score wouldn't make it safe for real use.
- Thirty questions or cases can reveal big failures, but can't prove reliability.
- Mac results don't show phone speed, memory, or battery use.

## Before anyone outside private testing can use this
Diagnosis and dose features need clinical and legal review. Every release requirement in the [safety policy](../safety-policy.md) must also be met.

## Results
Search comparison: not run yet.
