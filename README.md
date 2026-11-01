# Survival TR (working name)

A Turkish earthquake assistant prototype that works with no internet.

**Status: private test prototype. Do not use it in a real emergency.** Its content isn't expert-reviewed, its medicines are invented, and its LLM makes dangerous mistakes: in testing, Gemma-2-2B missed 7 of 14 invented emergencies. See [decision 002](docs/decisions/002-private-mac-prototype.md).

## How it works
```text
Question or scenario -> hybrid search (keywords + EmbeddingGemma) -> closest resources -> Gemma-2-2B -> answer + source text
Dose form -> ordinary code + a table of invented medicines
```
Everything runs on this Mac at `127.0.0.1`. There are no accounts, and nothing is logged. If the LLM fails, the app shows the closest source text.

## Run it
The models are in `.local/models` (2 GB, not in git). Use three terminals:
```bash
MODEL=embeddinggemma-300m-qat-Q8_0.gguf PORT=8098 ./scripts/start_model.sh --embedding -c 2048
./scripts/start_model.sh --reasoning off
python3 prototype/server.py    # after running once: python3 prototype/pack.py
```
Open http://127.0.0.1:8080. Together the models use about 2 GB of memory and may slow an 8 GB Mac. Press Ctrl+C in each terminal to stop them.

## Test it
```bash
python3 -m unittest discover -s tests
```
These tests use fake models and need no model running. Model scoring scripts are in `evals/phase2`.

## Project history
Phase 1 found that [Kumru-2B wasn't safe for rewriting instructions](docs/decisions/001-phase1-kumru-writer.md). Phase 2 is the current prototype.

## Release rule
No public release without expert review of the content, a passing safety evaluation, and legal review. A danger check and fixed emergency cards must also return first. See the [safety policy](docs/safety-policy.md).
