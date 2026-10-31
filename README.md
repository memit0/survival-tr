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
Phase 0: agree on the safety rules in [docs/safety-policy.md](docs/safety-policy.md), find a medical or emergency
reviewer, and check content rights.

Next is Phase 1: test whether Kumru-2B can simplify Turkish passages without changing their meaning. That test decides
whether the app uses Kumru, tries a backup model, or stays search-only.

## Release rule
No public release without expert review of the content, a passing safety evaluation, and legal review.
