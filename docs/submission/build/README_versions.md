# Two builds of the same paper

Both are generated from the same content and carry every correction; they differ only in how much
prose was compressed to meet the page limit.

| file | body | total | use |
|---|---|---|---|
| `main.tex` / `main.pdf` | **exactly 10 pages** (ends on §10 Contribution) | 16 | **the submission.** Meets the venue's "no longer than 10 pages, unlimited appendices" rule with no argument needed. |
| `main_full.tex` / `main_full.pdf` | 11 pages | 17 | the uncompressed version — fuller abstract, itemised Reliability/Limitations/Contribution lists, the display equation in §2, and the longer conclusion. Over the limit; keep for reading, revision, and any venue without a 10-page body cap. |

**Nothing scientific differs between them.** Same claims, same numbers, same figure, same seven
citations, same five appendices, zero placeholders in either. The trimmed build compresses prose
only: bullet lists become paragraphs, the abstract drops the position/hysteresis clauses, the
§2 architecture display becomes a sentence, and Acknowledgments moves to back-matter.

Rebuild either with two `pdflatex` passes from this directory. Both are CP1251 — use `grep -a`,
and never save them from a UTF-8 editor.
