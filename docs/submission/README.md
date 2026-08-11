# Mirror of the submission working files

**These are copies.** The live files are in `../../../files/paper_submission/`, which sits
**outside any git repository** — so if that directory is lost, these are the only copies. Mirror
kept because at 18:20 on submission day they were the most valuable artifacts in the project and
had no backup at all.

| file | what it is |
|---|---|
| `SUBMISSION_STATE.md` | **read first** — deadline, compliance checklist, placeholders by owner, template traps already solved, schedule |
| `RESULTS_HANDOFF.md` | every number with its scope attached and its ledger row; the *only* content source for the draft |
| `WRITING_PACK.md` | argument spine, final prose for the six hard paragraphs, anti-overclaim table, writer traps, paste-ready OpenReview fields, related work, novelty boundaries, reviewer objections, build mechanics |
| `APPENDIX_B_RETRACTIONS.md` | the full withdrawal/amendment list, written out |
| `CLAIM_INVENTORY.md` | 21 claims with presuppositions, experiment links, gates; Part 0 anti-2D-PCA; Part 6 prohibitions |
| `REVIEW_QUEUE.md` | the seven framing choices no author has seen — the actual remaining work |
| `build/` | **a self-contained, buildable copy of the paper** — see below |

## `build/` — the paper, buildable from a fresh clone

Added 2026-08-11 19:10. Before this, the repo held the `.tex` source but none of the assets it
needs, so nobody cloning it could compile the PDF or check the page limit.

| file | note |
|---|---|
| `main.tex` | the draft source. **CP1251 — use `grep -a`; never open with a UTF-8 tool and save.** Replaces the old `main.tex.cp1251` snapshot, which was byte-identical and has been deleted rather than left to drift. |
| `zapiski.cls` | the venue class. **Also CP1251**, and CRLF. |
| `pic/model.png`, `pic/symbols_fscore_on_threshold.csv` | the only two assets the source references |
| `main.pdf` | the compiled artifact, for reading without a TeX install |
| `instructions.md` | the venue's own submission rules — the source of the compliance checklist |

Build and verify:

```sh
cd docs/submission/build && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

**Verified 2026-08-11 19:10 from a clean copy of this directory alone**: exit 0, 10 pages,
266993 bytes — byte-for-byte the size of the committed `main.pdf`, 0 LaTeX errors, 0 undefined
citations.

**The six `LaTeX Font Warning: Font shape 'T2A/...' undefined` lines are expected and benign.**
They are Cyrillic-encoding font-shape substitutions from the venue class, not missing references;
`grep undefined` on the log matches them and reads alarming. Do not re-diagnose them.

**If you restore the live tree from here:** copy `build/*` back to
`files/paper_submission/draft/`, `instructions.md` to `files/paper_submission/`, and keep both
`.tex` and `.cls` in CP1251.
