# Real-article evaluation of Turkish offline search (2026-11-01)

**Verdict: not safe for emergency use yet.** The pipeline works offline and is fast, but on life-threatening questions the first excerpt was usable about one time in three, and the system cannot say "I don't know".

The corpus, downloaded files and question labels stay out of git because reuse rights for these sources are unconfirmed (see `docs/safety-policy.md`). They exist only on the author's machine, so these numbers cannot be reproduced from the repository alone. No code changed for this evaluation.

## What was tested

- **Corpus:** 33 articles, 408 passages, from 6 real documents, indexed whole, with nothing hand-picked.
- **Questions:** 90 (69 answerable, 21 with no answer in the corpus). 56 of the answerable ones are life-critical. Labels were written before any search output was seen and checked as exact quotes.
- **Process:** dev split (48) was used to choose the method and cutoff, then both were frozen. Test split (42) was run afterwards. Frozen choice: dense search, cutoff 0.82.

| Source | Year | Note |
|---|---|---|
| [AFAD Afet ve Acil Durumlar Eğitim Kiti](https://reyhanli.meb.gov.tr/meb_iys_dosyalar/2016_02/26033958_afad5.pdf) | 2015 | MEB mirror; guessed afad.gov.tr page URLs returned 404 |
| [Türk Kızılayı İlk Yardım Cep Kitabı](https://fethiye.meb.gov.tr/meb_iys_dosyalar/2018_11/01090747_5.2_ilkyardimcepkitabi.pdf) | 2015 | MEB mirror |
| [AKUT Temel İlk Yardım](https://ilkyardim.akut.org.tr/wp-content/uploads/2020/09/AKUT-Temel-Ilk-Yardim-Egitim-Kitapcigi.pdf) | 2019 | Only source with CPR steps |
| [TOTBİD Deprem Sonrası Hastalara Yaklaşım](https://www.totbid.org.tr/uploads/files/deprem_kitapcik.pdf) | 2021 | Written for orthopedic doctors, not the public |
| [TRT Haber: AFAD ilk 6 saat](https://www.trthaber.com/haber/turkiye/afad-deprem-sonrasi-yapilmasi-gerekenleri-paylasti-560920.html) | 2021 | News report quoting AFAD |
| [Memurlar.net: AFAD bilgilendirmesi](https://www.memurlar.net/haber/935256/deprem-aninda-ve-sonrasinda-neler-yapilmali.html) | 2020 | News report; recommends "hayat üçgeni" |

## Results

Frozen dense search, held-out test split. Ranges are 95% bootstrap intervals, and they are wide because the set is small.

| Measure | Result |
|---|---|
| Right article ranked first | 68% (21/31) [53–81] |
| Right article in top 3 | 90% (28/31) [79–100] |
| Labeled answer in first excerpt | 29% (9/31) [12–48] |
| Labeled answer in any of the 3 shown excerpts | 65% (20/31) |
| Labeled answer in top 20 passages | 81% (25/31) [68–93] |
| No-answer questions wrongly answered, no cutoff | 100% (11/11) |
| No-answer questions wrongly answered, cutoff 0.82 | 64% (7/11), and 3% (1/31) of answerable questions wrongly rejected |

Excerpt labels are narrow, so 29% is a floor. I therefore read each first excerpt and judged it:

| First excerpt judged by reading | Usable | Partial | Wrong or unusable |
|---|---|---|---|
| All 69 answerable | 24 (35%) | 14 (20%) | 31 (45%) |
| 56 life-critical | 18 (32%) | 12 (21%) | 26 (46%) |
| Test only (31) | 11 (35%) | 5 (16%) | 15 (48%) |
| Natural wording (44) | 20 (45%) | 10 (23%) | 14 (32%) |
| Paraphrased, no shared words (14) | 1 (7%) | 3 (21%) | 10 (71%) |
| No Turkish letters (10) | 2 (20%) | 1 (10%) | 7 (70%) |

### Resource use

Search takes about 15 ms once loaded (P50). A cold start takes 5.8 s and about 600 MB peak memory. The model is 466 MB, the index 1.6 MB, and indexing took 16 s. `offline-check` passed. The project's tests: 14 passed, 1 skipped.

## What went wrong

1. **It finds the topic but often not the instruction.** Passages split on topic rather than on the action someone needs, so symptoms or definitions rank above the steps.
   - "Karbonmonoksit zehirlenmesinde ne yapılır?" shows symptoms only.
   - "Camla derin kesildi, oluk oluk kan akıyor" shows the definition of bleeding.
2. **Some wrong results are dangerous if followed or waste critical time.**
   - "Bebeğin boğazına cisim kaçtı" shows eye-injury steps.
   - "Yanığa diş macunu sürebilir miyim?" shows heat-stroke steps.
   - "Kolu kopan kişiye ne yapmalıyım?" shows the recovery position.
   - "Kapalı yerde mangal yaktık herkesin başı dönüyor" shows driving advice.
   - "enkaz altindayim ne yapmaliyim" shows a physician's table.
   - "sarsinti sirasinda merdivenden inip kacayim mi" shows a bibliography.
3. **It cannot say "no answer".** Top scores for answerable and unanswerable questions overlap (dev: 0.81–0.91 against 0.78–0.87). "Python'da liste nasıl sıralanır?" returned choking steps, and a prompt-injection text returned first-aid chapter titles.
4. **One news article dominates.** Memurlar is 3% of the text but ranked first on 36 of 90 questions. Its long, narrative passages match many queries. "Hayat üçgeni" appeared in the first excerpt on 11 of 90 questions and in the shown excerpts on 15 of 90.
5. **Doctor-only content reaches the public.** "Enkazdan çıkarılan kişiye ne kadar sıvı verilir?" retrieves the physician guide, which contains IV-fluid guidance. The safety policy forbids dosing unless it comes from a reviewed passage.
6. **Critical topics are missing, and nothing routes to fixed cards.** The corpus has nothing on water purification, generators, childbirth or shelter. A self-harm query returned generic earthquake text, with no crisis response.

## Method comparison

The differences are within noise, so none of this supports switching methods.

| Right article first | Dense | Hybrid | BM25 |
|---|---|---|---|
| Dev | 79% | 71% | 53% |
| Test | 68% | 77% | 58% |

Hybrid beat dense on test but lost on dev, and the intervals overlap. Picking hybrid now would be choosing a winner after seeing the test.

## Non-blind experiment: removing the problem sources

Memurlar and TOTBİD were removed and title-only passages fixed. This used the already-seen test set, so treat it as a direction, not a score. Labels also change, because answers in the removed sources no longer count.

| Measure | Before | After |
|---|---|---|
| "Hayat üçgeni" shown (dev + test) | 15 of 90 | 0 of 89 |
| Physician guide ranked first | 6 of 90 | 0 of 89 |
| Right article first, dev | 79% | 68% |
| Right article first, test | 68% | 77% |

Removal clears those two exposures but does not clearly improve accuracy. Retrieval quality still needs its own fix.

## Dev experiment: embedding passage text only

Hypothesis: titles and headings make passages from one article look alike, so removing them from the embedding input should help pick the right passage. Passage boundaries stayed identical, and the test split was not run. The change was made in a copy of the prepared data, not in the code.

| Dense search, dev (38 answerable) | With title and heading | Text only |
|---|---|---|
| Right article ranked first | 79% (30) | 74% (28) |
| Right article in top 3 | 90% (34) | 87% (33) |
| Recall@3 | 0.78 | 0.66 |
| Labeled answer in first excerpt | 29% (11) | 34% (13) |
| Labeled answer in any of 3 shown excerpts | 42% (16) | 37% (14) |
| First excerpt judged usable | 13 | 15 |
| First excerpt judged wrong or unusable | 16 | 14 |
| Memurlar ranked first (of 48) | 22 | 8 |
| "Hayat üçgeni" in first excerpt (of 48) | 6 | 6 |

**Result: mixed, and not adopted.** The paired change in right-article-first is -5 points, with an interval of -19 to +9, so it could be noise in either direction. The usable counts were judged knowing which version was which, so they are weak evidence.

- **Better:** action steps replaced symptoms or fragments for carbon monoxide, burns and flooding, and the news article's dominance fell sharply.
- **Worse:** very short passages now win on little content. "kan durmuyor ne yapayim" shows only "Ağrı / Kanama / Yara kenarının ayrılması", and the seizure question now shows dislocation steps.
- **Unchanged:** the no-answer problem. The best cutoff still accepts 70% of unanswerable questions.
- **Hybrid fell from 71% to 61%**, but this run also removed titles from BM25, so it is not a clean test of the embedding change.

## Content conflicts for a medical or emergency reviewer

These are conflicts in the sources, and I did not settle them. They are not retrieval problems, and search scores cannot detect them.

- **Hayat üçgeni or Çök-Kapan-Tutun.** Memurlar's 2020 report attributes "hayat üçgeni" to AFAD's site, while AFAD's 2015 kit and 2021 statement say [Çök-Kapan-Tutun](https://www.trthaber.com/haber/turkiye/afad-deprem-sonrasi-yapilmasi-gerekenleri-paylasti-560920.html). [Evrim Ağacı](https://evrimagaci.org/hayat-ucgeni-vs-cok-kapan-tutun-deprem-sirasinda-hangi-yontem-uygulanmali-9505) argues against the triangle, but [AKUT's founder has criticized Çök-Kapan-Tutun](https://www.diken.com.tr/?p=1272920). I found no official AFAD text explicitly rejecting the triangle.
- **Loosening a tourniquet.** AKUT (2019) says to loosen every 15–20 minutes. [First-aid training material citing 2020 ILCOR guidance](https://pros-blog.padi.com/?p=378634) says only a health professional should remove one. I did not reach the ILCOR text itself or current Turkish guidance.
- **Moving a fracture.** AKUT describes gently pulling a broken leg into line. Kızılay says not to try to put a fracture back in place.

## Limits of this evaluation

- I wrote the questions and labels after reading the sources, so wording may be closer to the text than real users' questions. Real people may do worse, especially given the paraphrase result.
- All judging is by one AI reader. No Turkish speaker or medical expert reviewed anything. "Usable" means the excerpt matches the source's advice, not that the advice is medically correct.
- My PDF extraction is imperfect. A heading bug created title-only passages, which caused at least 2 of the misses (`burn-2`, `poison-bleach-1`). They were left in the frozen numbers so the test stayed blind. Bulleted lists and tables also lose some structure, as the TOTBİD table shows.
- Sources date from 2015–2021, and two are mirrors of the original PDFs.
- Search-result summaries were not trusted as evidence. One claimed that Kızılay bans tourniquets, but in the PDF that line only covers snake and scorpion bites.

## Local files (not in git)

These are in `reports/eq-2026-11-01/` on the author's machine.

| Path | Contents |
|---|---|
| `articles.jsonl`, `queries.initial.jsonl` | Frozen corpus and questions (SHA-256 of questions starts `fa877070`) |
| `articles.v2.jsonl`, `queries.v2.jsonl` | Non-blind experiment inputs |
| `runs/` | Every report and prediction |
| `scripts/` | Corpus and question builders, plus `judgements.py` with each judgement |
| `raw/` | Downloaded source files |
