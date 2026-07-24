"""Diagnostic: does the real Huginn tokenizer actually give one-token-per-digit
for this project's synthetic task prompts, and where does the multi-digit
answer split?

OWNER: Data+Analysis
STATUS: implemented 2026-07-24 -- one-off diagnostic, not a scored experiment.
    Not wired into cached()/results/ by design; resolves the uncertainty
    register item in project_plan.md §13 ("Digit tokenization... never
    verified for this tokenizer"). Needs only the tokenizer (CPU, no GPU,
    no model weights) -- much cheaper than a full extraction run.
TASK: check two things directly against the real tokenizer instead of
    assuming them: (1) does `" ".join(digits)` (what `make_count_ones_task`
    generates) tokenize as exactly one token per digit, matching that
    function's own docstring claim ("n_ops tokens, always")? Also checks
    the unspaced concatenation, since an earlier open question was whether
    inserting spaces between digits even matters here. (2) at what integer
    value does a first-token-argmax answer check (used by
    run_v6_correctness_probe.py and any accuracy check on count_ones/
    counting/three_scale) stop seeing the true answer as a single token --
    i.e. confirm the multi-digit-answer problem project_plan.md §9 already
    flags, with a live check instead of an assumption.
I/O: no args -> printed diagnostic only, nothing saved.

Run: uv run python -m scripts.diag_tokenization
"""

from __future__ import annotations

from transformers import AutoTokenizer

from traj_geom.constants import MODEL_ID, MODEL_REVISION


def main() -> None:
    """Print the two tokenization checks against the real tokenizer."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    print("--- (1) One-token-per-digit? (make_count_ones_task's own claim) ---")
    bits = "01010110101"
    for label, text in (("spaced", " ".join(bits)), ("unspaced", bits)):
        ids = tok.encode(text, add_special_tokens=False)
        decoded = [tok.decode([t]) for t in ids]
        ok = "MATCHES" if len(ids) == len(bits) else "MISMATCH"
        print(f"  {label:9s}: {len(bits)} digits -> {len(ids)} tokens [{ok}] {decoded}")

    print("\n--- (2) Where does a single-token answer check break? ---")
    for n in (0, 1, 5, 9, 10, 12, 20, 32):
        ids = tok.encode(f" {n}", add_special_tokens=False)
        decoded = [tok.decode([t]) for t in ids]
        flag = "single-token" if len(ids) == 1 else "MULTI-TOKEN, first-token-argmax wrong"
        print(f"  answer={n:3d}: {len(ids)} token(s) {decoded} -- {flag}")


if __name__ == "__main__":
    main()
