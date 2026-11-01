# Decision 001: Kumru as the answer writer (Phase 1)

**Result: fail.** Kumru-2B must not rewrite emergency instructions in this design.
**Status:** my reading of the Turkish still needs confirming by a Turkish-speaking reviewer. The next step awaits a decision.

## What was tested
- Model: Kumru-2B, Q4_K_M, community GGUF `icecubetr/Kumru-2B-GGUF`, on llama.cpp b12023 on an 8 GB Apple Silicon Mac, offline.
- Data: 20 short Turkish passages (72 sentences), hand-written from official guidance summaries. **Test data only, not expert-reviewed.**
- Pass rule: no output that the checker approves may change a number, lose a negation or drop a required warning.

| Round | Design | Passages passing the checker |
|---|---|---|
| 1, plain | Whole cited list, no grammar | 0 of 60 |
| 1, grammar | Whole cited list, with grammar | 3 of 60 |
| 2 | One sentence at a time, app adds the citation | 5 of 60 (99 of 213 steps) |
| 2, first sentence only | Rambling cut off, re-scored without new model runs | 179 of 213 sentences |

Speed was about 29-41 tokens per second on the Mac, which is not the limiting factor.

## Why it failed
Keeping only first sentences is the fairest case. Of the 179 approved sentences, 90 merely copy the source and simplify nothing.
The other approved sentences include real meaning changes the checker can't see:

| Source | Approved output | Problem |
|---|---|---|
| asla tatmayın (never taste) | asla koklamayın (never smell) | Wrong instruction |
| Hat aracınıza değiyorsa yangın yoksa araçtan inmeyin | Hat aracınıza değiyorsa, araçtan inmeyin | Drops the fire exception |
| Yangın yoksa ... | Yangın yoksa, araçtan inmeyin | Drops the power-line condition |
| evden en az 6 metre uzakta | evden -6 metre uzaklıkta | Garbles the distance |
| ev yapımı ve hasarlı konserveleri atın | hasarlı konserveleri atın | Keeps home-canned food |
| etiketini çıkarıp yıkayın | etiketini, yıkayıp atın | Wrong action |
| Şeker ve tuz eriyene kadar | Şeker suda eriyene kadar | Drops the salt |
| Soğuk su, buz torbası | Soğuk, buz torbası | Drops the water |
| su bulundurun | su/gıda stoğu bulundurun | Adds food |

Full sentences were worse: they looped, added emoji and invented instructions such as "Çalışıyorsa hemen kapatın" about a carbon monoxide alarm.

## What this does and doesn't show
- **Shows:** with these prompts and this build, rewriting isn't safe, and word-level checks can't catch these errors.
- **Doesn't show:** that Kumru is bad at Turkish overall. The test passages were already simple, leaving little to improve.
- **Not tested:** other builds, more prompt tuning, or Kumru's planned search role.
- **Design flaw found:** my round 1 example leaked into the outputs. Round 2 removed it, and round 1 is kept for the record.

## Reproduce
These scripts were removed from the current tree; run them from commit `f0e4dab`.

```bash
./scripts/start_model.sh                 # in one terminal
python3 evals/phase1/run.py              # round 1
python3 evals/phase1/run_sentences.py    # round 2
python3 evals/phase1/first_sentence.py   # re-score, no model needed
```
