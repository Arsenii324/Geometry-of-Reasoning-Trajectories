> **FROZEN 2026-08-05.** Archived under CLAUDE.md §6 (three live documents only).
> Not maintained. Superseded by `docs/UNDERSTANDING.md`, `claims_ledger.md`, `directions.md`.
> Statements here may be contradicted by later work — several assert things since disproven
> (e.g. "H1 was never tested": it was, and the test itself was invalid — see D56, D65).

# Verifiability & accountability practices

STATUS: written 2026-07-24, in response to an explicit request to formalize
this project's verification/accountability practices rather than continue
applying them ad hoc. This doc does three things: (1) states the actual
requirements (mandatory vs nice-to-have), (2) audits current state against
them, (3) records what was fixed and what's deliberately deferred.

This is a practices doc, not a data doc — it makes no experimental claims
of its own. For those, see `claims_ledger.md` (the primary artifact this
whole framework exists to support).

---

## 1. Requirements analysis

### 1.1 What is actually at risk, for this specific project

A small team, working in short GPU-bound bursts (Kaggle/DataSphere sessions),
across multiple AI-assisted sessions with no persistent memory between them,
producing docs and code faster than any one person reviews all of it. The
concrete failure modes already observed this session, not hypothetical ones:

- A finding gets recorded, then a bug is found in the code that produced it,
  and the finding is never revisited (the original coda-skip bug: two
  rounds needed before it was actually fixed, each round could have been
  the last if nobody re-checked).
- A number gets cited from a secondary source (a WebFetch summary of a
  paper) and is wrong in a way that matters for real arithmetic (the
  Blayney 2.81%/0.14% conflation this session — self-caught, but only
  because the number was about to be used in a real calculation, not
  because anyone was checking sources systematically).
- A statistic is computed correctly but its applicability conditions are
  silently violated by real data (the `_SPEARMAN_CRIT_P05` tie-handling gap
  found today — the table assumes no ties; real data ties; nobody had
  checked this in the ~20 places the table is used before today).
- Documentation drifts: five docs (`narrative.md`, `research_log.md`,
  `end_to_end_guide.md`, `guide.md`, `infra.md`) haven't been touched since
  2026-07-22, before the coda-skip fix's second round, before D11's
  refinement, before D12's two corrections, before D13/14/15, before the
  literature-verification pass. None of them say this.
- A doc-facing claim (README's "2.81% on question tokens") is now known
  imprecise (it's a per-example rate, not per-token) but the README itself
  hasn't been told.

### 1.2 Mandatory requirements

These are non-negotiable for the project not to actively mislead someone
relying on it:

- **M1 — Every experimental claim must carry its own verification status.**
  Not "verified once, forever" — a live tag (verified-live / trusted-cached /
  contradicted / unimplemented) plus the exact evidence and date. Already the
  strongest existing piece of this project's practice (`claims_ledger.md`'s
  entire design). Requirement: keep applying it, don't let it lapse for new
  work.
- **M2 — A caveat, once found, must be attached to the claim it caveats, not
  left in whichever document happened to be open when it was found.** A
  finding that only lives in a chat transcript or a single script's inline
  comment, with no pointer from the claim it affects, is not verifiable by
  anyone who starts from the claim.
- **M3 — Superseded documentation must say so, at the point a reader would
  encounter it, not just in a separate index.** An index that says "read
  this in order, newer stuff first" is necessary but not sufficient — a
  reader who jumps straight to `narrative.md` (a very plausible thing to do,
  it says "start here if you read only one thing" in `START_HERE.md`) needs
  to see the staleness warning *in that file*, not just in the index they
  may have skipped.
- **M4 — Every script must state its own verification status inline,** not
  only in an external ledger. A reader looking at `run_maxtask.py` should
  not need to cross-reference `claims_ledger.md` to learn its main result is
  confounded (D10) — the confound should be stated in the file itself.
- **M5 — Absence of evidence must be stated as absence, not omitted.**
  "Never independently verified" and "verified and correct" must be
  visually and textually distinct; a doc that's silent on a point should
  not be readable as an implicit "checked, fine."
- **M6 — No claim should be un-reproducible in principle.** Every
  number in `claims_ledger.md` should trace to a script + commit that a
  reader could actually re-run (this project's own `test_architecture_consistency.py`
  already partially enforces this for CSV/script pairing).

### 1.3 Nice-to-have (valuable, deliberately not built this pass, with reasons)

- **N1 — Fully automated cross-doc link/reference validation** (a script
  that parses every `claims_ledger.md` ID reference across all docs and
  confirms the ID exists, every commit hash cited still exists, etc.).
  Deferred: high effort for a small team's doc corpus size; manual
  discipline plus the mechanical checks below cover the highest-risk cases
  (CSV/script pairing) already.
- **N2 — A single machine-readable hypothesis-status table** (e.g. a YAML
  file with H1/H2/H3 status, auto-rendered into docs). Deferred:
  `project_plan.md` §1/§2 already serves this role in prose form reasonably
  well; converting to structured data is a real improvement but not
  urgent — the prose version is actively maintained and current.
  Revisit if the hypothesis set grows past H1/H2/H3 or gets a second
  consumer (e.g. an automated report generator).
- **N3 — Formal versioning of docs** (e.g. every doc gets a `SUPERSEDES:` /
  `SUPERSEDED-BY:` frontmatter field, checked by a test). Deferred: the
  banner approach in §3 below gets most of the value at a fraction of the
  mechanism; revisit if the doc count grows enough that manual banner
  maintenance becomes itself a source of drift.

---

## 2. Audit against the mandatory requirements (2026-07-24)

| Req | Status before this pass | Fix applied |
|---|---|---|
| M1 | Strong — `claims_ledger.md`'s tagging convention already does this well, applied consistently through D15. | None needed; kept applying it (D12/13/14/15 all follow it). |
| M2 | **Gap found**: `_SPEARMAN_CRIT_P05`'s tie-handling limitation existed in the code with no note anywhere, discovered only when it happened to matter for D15. | Documented directly in `correlate.py`'s own docstring (not just the ledger) — see commit `1dd23b4`. |
| M3 | **Gap found**: `narrative.md`, `research_log.md`, `end_to_end_guide.md`, `guide.md`, `infra.md` — all last touched 2026-07-22, predate every major correction since, no staleness marker in any of them. | Banners added, see §3. |
| M4 | **Gap found**: 9 scripts (`run_convergence/counting/dissociation/dissociation_multiinit/forceloop/homology/maxtask/phase/switch.py`) had `OWNER` but no `STATUS` — unlike every script touched this session. | Fixed, commit `7fe20d0` — each now cites the specific `claims_ledger.md` ID its result should be read alongside. |
| M5 | Mostly OK — `claims_ledger.md` already distinguishes "Trusted-cached" from "Verified-live" from "never independently verified" explicitly per-row. `project_plan.md`'s uncertainty register (§13) does the same for literature claims. | None needed beyond keeping the convention. |
| M6 | Mostly OK, enforced partly by `test_architecture_consistency.py`'s CSV/script pairing + staleness check. **One live gap**: README's "2.81%" figure (§3.3) doesn't cite a script because it's prose, not a computed claim — flagged, not a script-reproducibility gap. | Flagged in §3.3; not silently fixed (README is teammate-authored prose, established practice this session is to flag, not silently rewrite it). |

---

## 3. Fixes applied

### 3.1 Script STATUS lines (M4)
Done, commit `7fe20d0`. See that commit message for the full list.

### 3.2 Stale-document banners (M3)
Added a short, dated banner to the top of `narrative.md`, `research_log.md`,
`end_to_end_guide.md`, `guide.md`, `infra.md` — each states plainly that the
file predates this session's major corrections (coda-skip round 2, D11
refinement, D12's two corrections, D13/14/15, the literature-verification
pass) and points to `project_plan.md` + `claims_ledger.md` as current.
Content of these files is otherwise untouched — this is an addition, not a
rewrite, consistent with this session's established practice of not
silently editing teammates' authored prose.

### 3.3 README's "2.81%" imprecision (M6, partial)
`README.md`'s Results section states Blayney's rate as "до 2.81% на токенах
вопроса" (up to 2.81% on question tokens) — this project's own re-check
(`claims_ledger.md` B9, 2026-07-24) found 2.81% is Blayney's **per-example**
rate ("does any question token show the behavior"), not a per-token rate;
the correct per-token ceiling is 0.14%. This is currently only corrected in
`claims_ledger.md`/`power_and_preregistration.md`/`project_plan.md`, not in
the README teammates and the curator are most likely to actually read.
**Not fixed here** — same reasoning as every other README-touching decision
this session (an earlier explicit instruction: don't silently rewrite a
teammate's prose). Flagged here, in `project_plan.md`, and should be raised
directly with whoever owns README's Results section.

---

## 4. What this doc is not

Not a claim that the project is now fully debt-free — it is a record of
what was checked, what was found, and what was done about it, at one point
in time. Re-running this audit is cheap (grep the 5 doc mtimes, check for
scripts missing STATUS, re-check `_REVIEWED_NON_SUBSTANTIVE_CHANGES` isn't
hiding something real) and should happen again whenever a significant
finding lands, not treated as a one-time exercise.
