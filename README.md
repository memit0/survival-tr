# Survival TR (working name)

A free Turkish emergency assistant for phones that works with no internet.

**Status: early planning. This is not ready for use in a real emergency.**

## The idea
App code controls every step, and expert-reviewed Turkish content is the source of truth. A small on-device AI model
(Kumru-2B is the first candidate) only helps people understand and find that content. The app must stay useful with no
model running.

```text
Question -> Danger check -> reviewed emergency card + 112
                 | no match
                 v
              Search -> official passage -> optional simplified steps -> checker
```

## Current step
Phase 0 is still open: the safety rules in [docs/safety-policy.md](docs/safety-policy.md) need expert review, and we
still need a reviewer and content rights.

Phase 1 tested whether Kumru-2B can simplify Turkish passages without changing their meaning, and
[it failed](docs/decisions/001-phase1-kumru-writer.md). What Kumru does next is awaiting a decision.

## Release rule
No public release without expert review of the content, a passing safety evaluation, and legal review.
