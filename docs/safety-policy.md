# Safety policy

**Status: first draft, not yet reviewed by a medical or emergency expert or by a lawyer.**
Nothing here is final until a qualified reviewer approves it. Items marked REVIEW need a specific decision.

## Purpose
The app helps people in Turkey find trustworthy emergency guidance when they have no internet. It supports professional
emergency services and does not replace them.

## Core rules
1. **Reviewed content is the source of truth.** The AI model only helps people find and understand it.
2. **Life-threatening topics use fixed cards.** The model never writes this advice.
3. **Show the official text first.** Any simplified version appears after it and is clearly labelled as AI-written.
4. **Code has the final say.** The model can't overrule the danger check or the checker.
5. **When in doubt, don't guess.** If the checker fails, keep the official text. If nothing matches, say so.
6. **It must work without the model.** Cards and search must work on every supported phone.

## Severity levels
| Level | Meaning | What the app does | Model's role |
|---|---|---|---|
| `critical` | Could cause death or serious injury within minutes or hours | Shows a fixed card and 112 | Doesn't write the advice |
| `caution` | Could cause harm if done wrong | Shows the official passage | May simplify it, subject to the checker |
| `general` | Preparation and everyday information | Shows the official passage | May simplify it, subject to the checker |

REVIEW: confirm these levels, and assign each topic to one.

## Proposed critical topics
REVIEW: the expert should add, remove or rewrite these topics.

- Earthquake: during shaking, trapped under rubble, and aftershocks
- Severe bleeding
- Not breathing, or no signs of life
- Choking
- Carbon monoxide, and generators, grills or heaters used indoors
- Fire and smoke
- Floodwater and fallen power lines
- Serious burns
- Poisoning and suspected botulism
- Hypothermia and heat stroke
- Childbirth and emergencies involving infants
- Suicide or thoughts of self-harm

## What the app may do
- Show reviewed cards, passages and checklists.
- Simplify `caution` and `general` passages when every step cites its source and passes the checker.
- Point to 112 and show people how to get help.
- Use profile facts that the person confirmed, such as an allergy, to choose which reviewed warnings to show.

## What the app must not do
- Diagnose a condition.
- Calculate or suggest a medicine dose, unless that exact dose guidance is in a reviewed passage shown as written.
- Tell someone it is safe not to seek help.
- Invent sources, phone numbers, addresses, shelters or current conditions.
- Claim to know live information such as road status, aftershock forecasts or shelter availability.
- Contact anyone without the person's action. Calling 112 always needs their tap.
- Send questions or personal data off the phone.

## How the app words things
- Use short, direct, plain Turkish, with the most urgent step first.
- Be clear about the content's source and its last review date.
- Use honest wording when nothing is found. For example: "Bu konuda kitaplıkta güvenilir bir bilgi bulamadım."
- Don't add reassurance that the source doesn't support.
- Label simplified text. For example: "Yapay zekâ özeti. Asıl kaynak yukarıda."

REVIEW: a Turkish-speaking expert should approve all fixed wording.

## Prompt injection and misuse
- Questions, passages and model output are treated as untrusted text, and none of them can change these rules.
- "Ignore your rules" requests get the same handling as any other question.
- The app doesn't help with causing harm to people.

REVIEW: decide how to respond to dangerous requests, while still giving people safety information, such as what to do after a chemical exposure.

## Personal data
- There are no accounts, and no analytics by default.
- Profile data stays on the phone, uses the phone's file protection, and can be deleted in one step.
- Nothing is saved to the profile until the person confirms it.

REVIEW: legal check under Turkish data protection law (KVKK).

## Content rules
Every content item needs:
- A source, publisher, address, license and retrieval date
- A named reviewer and review date
- A severity level
- Any required warnings marked

Content with unclear reuse rights can't be included. AI-generated text can't count as expert review.

## Private prototype exception
**Status: applies only to the Mac prototype in [decision 002](decisions/002-private-mac-prototype.md). It is not expert-approved.**

This exception allows these features that the rules above forbid, so they can be tested:
- The LLM may answer any question, including `critical` topics, using the closest retrieved test resources.
- The LLM may suggest a possible diagnosis from those resources.
- The prototype has no danger check or fixed emergency cards.
- Code may calculate a dose from a table of invented medicines.

While this exception is in use:
- Use it only with the owner and testers who know it is a test, and only with invented scenarios.
- Don't use real medicine names or real health problems.
- Label all medical output: "Yalnızca test. Uygulamayın."
- Keep a fixed banner on screen: "Yalnızca test. Gerçek acil durumda 112'yi arayın."
- Show the source text with every LLM answer, and mark answers from weak matches as low match.
- Never let the LLM calculate or change a dose.

This exception doesn't change the rules for any release. A danger check and fixed emergency cards must be back in place before anyone outside private testing uses the app. Diagnosis and dose features also need clinical and legal review, and every release requirement below still applies.

## Release requirements
- [ ] A qualified expert has reviewed all content and fixed wording
- [ ] The safety evaluation has 0 harmful answers and 0 made-up sources on the exact release build
- [ ] Every critical test question reaches its card
- [ ] Failure tests keep the official text on screen
- [ ] The app works fully in airplane mode on the test phones
- [ ] Legal review is complete, including liability, KVKK and medical device rules
- [ ] A process exists for reporting errors and updating content

## Open questions
1. Who will be the medical or emergency reviewer?
2. Which sources can we reuse, and do we have that in writing?
3. How should guidance differ for children, older adults and pregnancy?
4. How often must content be reviewed, and what should the app show when content is overdue?
5. Could the app legally count as a medical device?
