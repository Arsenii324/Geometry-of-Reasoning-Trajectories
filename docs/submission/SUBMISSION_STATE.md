> **Lost? `docs/INDEX.md` indexes every document, tool, data location and external path, with a read order.**

# SUBMISSION STATE — read this first after a context compaction

*Written 2026-08-11 17:40, **updated 18:15**. The deadline is **21:00 Moscow time** and it is an
**upload-complete-with-all-formalities** deadline, not a "stop working" deadline.*

---

## 0. The single highest-risk item

**"All authors must be registered on OpenReview and added to the submission."** OpenReview
profile creation can require email verification and sometimes manual approval — hours or days,
not minutes. **If a co-author lacks a profile this is a hard blocker.** The supervisor was asked
to verify this at 17:40. *If it is still unverified, raise it again immediately; nothing else
matters if this blocks.*

## 1. The deliverable, and its current state

`files/paper_submission/draft/` — `main.tex`, `main.pdf`, `zapiski.cls`, `pic/`. **Untracked, outside
the research repo, nothing committed.**

- **8 pages**, exit 0, **0 LaTeX errors, 0 undefined citations**, 0 overfull boxes > 20pt.
- `main.tex` is **CP1251** (`file` reports ISO-8859). **Plain `grep` treats it as binary — use
  `grep -a`.** This caused one false alarm already ("all numbers missing from draft" — they were
  all present).
- Content came from `RESULTS_HANDOFF.md` **only**. **That file is now current to 18:00 and
  carries A1–A11, so the compiled PDF is behind it by four results (D198, D199, D200 and the
  A7 amendment). Regenerating the draft from the handoff is a required step, not an optional
  refresh.**
- **IT IS NOT REVIEWED. Do not treat it as ready.** Every sentence in it derives from
  `RESULTS_HANDOFF.md`, which I wrote. **The authors have not read the content, and no claim in
  the PDF has been discussed with them.** It compiles and is compliant; that is a *floor*, not
  readiness. **A post-compaction pass must not optimise toward upload.** The remaining work is
  surfacing claims for the authors to check and challenge — errors found now are cheap, errors
  found by a reviewer are not.

### Template traps, already solved — do not re-break

- `zapiski.cls` is **CP1251**; the distributed `template.tex` is **UTF-8**, contradicting its own
  README. Writing the `.tex` in CP1251 as the README requires makes the Russian masthead render
  correctly **with no `\renewcommand` overrides**. The three overrides in the distributed
  template exist only to bridge its own mismatch and have been deleted. Do not restore them.
- The shipped template places the required **Russian title/authors/abstract/keywords block after
  `\end{document}`**, so it never typesets. It has been moved *before* `\end{document}`.
- `\begin{abstract}` after `\maketitle` is **silently dropped** by `amsart` — the Russian abstract
  vanished with a clean exit 0. It is now a formatted block.
- `\and` in `\keywords` renders the literal English word "and" between Russian terms. Now
  comma-separated.

## 2. Compliance checklist (from `instructions.md`)

- [ ] ≤ 10 pages main body (currently 8), unlimited appendices — **OK**
- [ ] Not anonymised — **OK**
- [ ] Full author list, **all authors registered on OpenReview** — **BLOCKED ON USER**
- [ ] Mentor(s) listed — name present (*Serguei Barannikov*), **affiliation is placeholder**
- [ ] **Contribution section, one entry per team member** — *stated twice in the instructions*;
      structure present, **entries are placeholder**

## 3. Placeholders, by who can clear them

**USER (critical path, needed by ~19:30):**
- author names + affiliations — `main.tex` line ~44
- mentor affiliation — line ~47
- **Contribution entries** — lines ~274–282
- Acknowledgments — line ~288
- Russian names in the metadata block — line ~376; confirm spelling *Сергей Баранников*

**ME (can be done any time):**
- Appendix B retraction list — content is **already written** in
  `files/paper_submission/APPENDIX_B_RETRACTIONS.md`; drop into `main.tex` line ~327 at regeneration.

## 3a. THE CLAIM INVENTORY — the primary writing source

**`files/paper_submission/CLAIM_INVENTORY.md`** (mirrored to `docs/submission/`). 21 claims,
6.6k words. Per claim: strongest defensible form, safe form where they differ, **presuppositions**,
the **exact experiment (kernel path + DataSphere job id, all 19 verified against the ledger)**,
the **gates that passed**, and what would break it. Plus Part 0 (the methodological case against
visual 2D PCA), Part 5 (presupposition index — state these once, early), Part 6 (the 14 things
that must not be claimed).

**Write from this.** `RESULTS_HANDOFF.md` is the prose-facing subset; the inventory is the
complete one and carries the links and caveats the handoff compresses away.

## 3b. The writing pack — read it before writing any prose

**`files/paper_submission/WRITING_PACK.md`** holds what `RESULTS_HANDOFF.md` deliberately does
not: the argument spine, **final prose for the six paragraphs that are hard to write without full
context**, an **anti-overclaim table** (may say / may not say, every right-hand entry a sentence I
actually had to correct), the traps a compacted writer falls into, ready-to-paste OpenReview
fields (title, TL;DR, keywords, abstract), figure candidates with their data paths, and the
contingency for the one unresolved job.

**If you are writing and short on context, read §C (anti-overclaim) and §D (traps) first.** They
are the parts that prevent a wrong sentence rather than a missing one.

## 3c. The citation defect found at 18:20 — RESOLVED 19:25, do not re-do

The draft had declared 6 `\bibitem`s and contained exactly one `\cite`; LaTeX's "0 undefined
citations" check runs the other way, so a green compile hid it. **Fixed and verified by counting:
6 `\bibitem`s, 6 `\cite` commands, keys one-to-one.** `WRITING_PACK.md` §H still records what each
reference is for, which remains useful when writing.

**Also verified at 19:25, so a later pass need not:** `epiphenomenon` appears once in the draft
and its use is the *negated* one required by `WRITING_PACK.md` §B1 — *"That is not evidence that
the regime is a dynamical epiphenomenon"*. `CLAIM_INVENTORY.md` Part 6 item 1 bans the assertion,
not the word. Do not "fix" this sentence. The other Part 6 phrasings ("orthogonal", "H2 is
refuted", "not linearly decodable") return **0** matches.

**Still true and still open: the draft contains 0 `figure` environments.**

## 4. Schedule, backwards from 21:00

**The earlier hour-by-hour schedule is withdrawn.** It optimised for shipping, and the authors
have said that is the wrong target — the work is incremental and the content is unvetted.

What remains, in dependency order rather than by clock:

| what | who | note |
|---|---|---|
| **surface the paper's claims for author review** | me → user | the real bottleneck; nothing in the PDF has been checked by an author |
| Contribution entries | user | the one compliance item still missing |
| corrections arising from that review | me | expect these; they are the point |
| final compile + encoding + page check | me | mechanical, ~5 min, do last |
| OpenReview form + upload | user | fields are paste-ready in `WRITING_PACK.md` §E, **also unreviewed** |

*Registration is confirmed. Authors: Varaksin, Shiianov, Sverdlov; mentor Barannikov (Skoltech).*

## 5. Behavioural commitments made at 17:35 — keep them

- **No further job launches**, regardless of what lands.
- Results landing **before 19:00** get folded in; after that, **ledger only**.
- **If a job errors: record it and move on. No diagnosis sessions.** Three of those today cost
  20–40 minutes each.
- Do not let result-chasing jeopardise the document.

## 6. Jobs — three landed and recorded, one outstanding (as of 18:15)

| exp | outcome |
|---|---|
| **A52** `ds_transrot` | **LANDED → D198.** The regime does not exist yet when the answer is decided: early separation **0.000**, tail separation **0.565**, onset dated between unrolls 12 and 16. Reframes D148/D176. |
| **A50** `ds_sysprompt` | **LANDED → D199.** The system turn buys **1 item in 24** over the same text in the user turn (p = 1.0). The one qualified exception is *"reason, then write `Answer:`"* — 0.167 vs 0.000, p = 0.125 at n = 24. |
| **A49** `ds_periods` | **LANDED → D200.** Their word problem does not orbit at **any** period; its non-DC power is **2.096** against 665–3139 for rotating prompts. Our detector validated to **4e-08**. |
| **A47** `ds_h0inject` | **STILL EXECUTING** (`bt1vuemjmeg0onc52shp`), 3rd attempt, 18/18 items surviving, ~4/18 done at 18:06. **See `WRITING_PACK.md` §G for the contingency — if it does not land by 19:00, omit it entirely.** |

**Procedure on landing:** `datasphere project job download-files --id <id>` from the kernel dir →
**check the P1 gate first, not the headline** → record a ledger row → update `RESULTS_HANDOFF.md`
if before 19:00.

*`job attach --id` is safe on a running job (verified: 166 jobs before, 166 after, status
unchanged) but **does not replay a job-side traceback** on an ERRORed job. And note RC9: a
platform ERROR is often the kernel refusing on its own gate, with no traceback ever raised.*

## 7. Research state at compaction

- `docs/claims_ledger.md` — **200 rows**. `docs/UNDERSTANDING.md` §1.0-today is the day's
  verification surface. `docs/directions.md` §M–§P carry the queue, the pre-registered
  predictions, the failure-cause recap and the resumption state.
  **`docs/WORKING_KNOWLEDGE.md`** is the tacit layer (instrument behaviour, load-bearing vs
  incidental numbers, fragility map, read-order, failure signatures).
- git `a14810a`, suite **616 passed**. **Gate every push on a green suite** — I pushed red twice
  today by running pytest and then committing unconditionally.
- Next free experiment number: **A53** (`scripts/experiment_registry.py` exits nonzero on a
  collision — two collisions happened today).

## 8. The emphasis decision, left open deliberately

Three separable contributions; the draft presents them in parallel with **no headline claim**:
1. **the regime line** (research contribution — novel, causal at two levels, contradicts a
   published attribution, survived a scope test),
2. **the measurement line** (methodological — and what makes the rest trustworthy under the
   "reliability of results" review criterion),
3. **H2's reframing** (hypothesis verdict — architecturally untestable per-position on Huginn).

A single unifying sentence if one is wanted: *we set out to measure the geometry of latent
reasoning, found that most of what we were measuring was output format, and fixing that changed
what the geometry results mean.*
