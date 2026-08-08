"""Answer probability as a function of recurrent depth, via teacher forcing.

OWNER: Data+Analysis
STATUS: implemented 2026-07-25. Needs a GPU to run; the estimator logic below
    is unit-tested against synthetic logits without one.
TASK: for one prompt, return log P(gold answer | r unrolls) for every r, so
    that "how much depth does this instance need?" becomes a measured number
    rather than a proxy.

WHY THIS METRIC, AND WHY NOT THE OBVIOUS ALTERNATIVES
  The hypothesis is that accuracy on an instance is near-zero until the model
  has unrolled enough times to actually carry out the computation, then rises
  sharply -- a task-dependent threshold r*. If r* scales with difficulty, that
  is a direct, behavioural statement of "latent depth encodes reasoning
  depth", with none of the geometry that docs/rigor_audit.md just dismantled.

  Three candidate measurements, and why this one:

  (a) GREEDY GENERATION per unroll. Correct but expensive (one generation per
      r per prompt) and binary, so it throws away all graded information --
      exactly the resolution needed to locate a threshold.

  (b) ARGMAX OF THE FIRST ANSWER TOKEN (what run_v6_correctness_probe.py
      does). Cheap, but the first token is not the answer. For a negative
      value the leading token is just "-", shared by every negative number,
      and for any answer >= 10 it is only the leading digit. That script
      records `is_single_token_answer` precisely so those rows are not scored
      as if fully checked -- and on the existing 24-row run only 13 rows
      qualify, of which the 4 that are ever "correct" all have target 0 and
      all become correct at unroll 1. That pattern is a prior on emitting
      zero, not computation.

  (c) TEACHER-FORCED SEQUENCE LOG-PROBABILITY (this module). Append the gold
      answer to the prompt and read, at each answer position, the probability
      the model assigns to the token that actually belongs there. Summing
      gives an exact log P(answer), multi-token answers included, with no
      generation at all. It is continuous, so a threshold is locatable; it is
      exact for negative and multi-digit answers, which is where (b) breaks;
      and because every answer position is scored in the SAME forward pass,
      one pass yields the entire r-curve.

  THE TOKENIZATION TRAP THIS INHERITS. The prompts end "...A:" with no
  trailing space, so the model's actual continuation is space-prefixed (" 2",
  not "2"). A prior version of the v6 probe scored the bare token and found
  correct_at_step = -1 for every single row as a direct artifact.

  BUT SCORING ONLY THE SPACE-PREFIXED FORM IS ALSO WRONG. The answer is a
  STRING; the model can express it through more than one token sequence, and
  P(string) is the sum over all of them. Verified against the pinned
  tokenizer -- both forms exist as distinct tokens::

      '2'   -> [50]        ' 2'  -> [402]
      '64'  -> [54, 52]    ' 64' -> [893, 52]
      '-3'  -> [45, 51]    ' -3' -> [410, 51]

  Scoring only " 2" therefore UNDERSTATES P(answer) by whatever mass sits on
  the bare form. `scripts/diag_v6_token_gap.py` measured that split once (' 2'
  ~0.065 against '2' ~0.0023, a ~28x ratio, so ~3.5% of the mass missed
  => ~0.034 nats) but at one position for one prompt; it is not safe to
  assume that holds everywhere.

  So `depth_curve` MARGINALISES: it scores every plausible surface form and
  combines them with log-sum-exp. Conveniently the variants have the SAME
  token length -- they differ only in the first token -- so this costs one
  extra forward pass per variant and no positional bookkeeping.

  Note this does not fully cancel in the margin even though both gold and
  distractor get the same treatment: the two answers can have different
  bare-vs-prefixed splits. Marginalising is the fix; relying on cancellation
  is not.

  THE CONTROL. A rising log P(gold) curve on its own proves little: the model
  may simply be getting more confident about everything, or the answer string
  may be intrinsically likely (again: "0"). `depth_curve` therefore also
  scores a DISTRACTOR of the same token length, and the reported quantity is
  the MARGIN log P(gold) - log P(distractor). A margin threshold cannot be
  produced by a uniform confidence increase, and it cannot be produced by a
  prior toward a particular answer string, because the prior lifts both.

I/O: depth_curve(model, tok, prompt, answer, distractor, max_r)
    -> dict with per-r log-probs for gold and distractor, and the margin.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# torch is imported lazily, inside `_teacher_forced_logp` -- the ONLY function here
# that touches it. It lives in the optional `model` extra, and importing it at module
# level made `from traj_geom.eval_depth import threshold_depth` fail without it, even
# though `threshold_depth`, `_logsumexp`, `answer_surface_forms` and `answer_token_ids`
# are pure. That turned a missing optional dependency into a pytest COLLECTION error,
# which aborts the whole run rather than skipping one module. `scripts/_common.py`
# already defers the model import the same way.

__all__ = ["answer_token_ids", "answer_surface_forms", "depth_curve", "threshold_depth"]

# Surface forms an answer can take right after "A:". Ordered most- to
# least-likely; the leading-space form dominates but is not the only one.
_SURFACE_PREFIXES = (" ", "")


def answer_token_ids(tok: Any, answer: Any) -> list[int]:
    """Token ids of the answer's DOMINANT surface form (space-prefixed).

    Kept for callers that need a single canonical length (e.g. matching a
    distractor's token count). For probability mass use `answer_surface_forms`
    and marginalise -- this form alone understates P(answer).

    Args:
        tok: The Huginn tokenizer.
        answer: The gold answer; stringified.

    Returns:
        Token ids of ``" " + str(answer)``. The leading space is not
        cosmetic -- see this module's docstring.
    """
    return list(tok(" " + str(answer), add_special_tokens=False).input_ids)


def answer_surface_forms(tok: Any, answer: Any) -> list[tuple[str, list[int]]]:
    """Every plausible way the model could spell ``answer`` at this position.

    P(answer) is a property of the STRING, not of one tokenization, so the
    depth curve must sum over these rather than pick one.

    Args:
        tok: The Huginn tokenizer.
        answer: The gold answer; stringified.

    Returns:
        ``[(surface_form, token_ids), ...]``, de-duplicated by token ids and
        ordered with the space-prefixed form first. Empty tokenizations are
        dropped.
    """
    out: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for prefix in _SURFACE_PREFIXES:
        form = prefix + str(answer)
        ids = list(tok(form, add_special_tokens=False).input_ids)
        key = tuple(ids)
        if ids and key not in seen:
            seen.add(key)
            out.append((form, ids))
    return out


def depth_curve(
    model: Any,
    tok: Any,
    prompt: str,
    answer: Any,
    distractor: Any,
    max_r: int = 64,
) -> dict[str, np.ndarray]:
    """Teacher-forced log P(answer) at every recurrent depth r = 1..max_r.

    Runs ONE forward pass per candidate over ``prompt + candidate``, hooking
    the recurrent core so that every unroll's hidden state is converted to
    logits by the model's own coda/head. Because teacher forcing scores all
    answer positions simultaneously, a single pass yields the whole curve.

    Args:
        model: Loaded Huginn model.
        tok: Matching tokenizer.
        prompt: The question text, ending in "A:" with no trailing space.
        answer: Gold answer.
        distractor: A wrong answer, ideally with the same token length as the
            gold one so the two log-probs are comparable term by term.
        max_r: Number of recurrent unrolls to sweep.

    Returns:
        Dict with ``r`` (1..max_r), ``logp_gold``, ``logp_distractor`` and
        ``margin`` (gold minus distractor), each of length max_r.

    Raises:
        ValueError: if either candidate tokenizes to zero tokens.
    """
    curves = {}
    detail: dict[str, np.ndarray] = {}
    for name, cand in (("gold", answer), ("distractor", distractor)):
        forms = answer_surface_forms(tok, cand)
        if not forms:
            raise ValueError(f"{name} answer {cand!r} tokenized to nothing")
        per_form = np.stack(
            [_teacher_forced_logp(model, tok, prompt, ids, max_r) for _, ids in forms]
        )
        for (form, _), row in zip(forms, per_form, strict=True):
            detail[f"logp_{name}[{form!r}]"] = row
        # log P(string) = log sum_forms P(form): marginalise, do not pick one
        curves[name] = _logsumexp(per_form, axis=0)

    return {
        "r": np.arange(1, max_r + 1),
        "logp_gold": curves["gold"],
        "logp_distractor": curves["distractor"],
        "margin": curves["gold"] - curves["distractor"],
        **detail,
    }


def _logsumexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable log-sum-exp (log-probs are large and negative)."""
    m = np.nanmax(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.nansum(np.exp(a - m), axis=axis))


def _teacher_forced_logp(
    model: Any, tok: Any, prompt: str, ids_ans: list[int], max_r: int
) -> np.ndarray:
    """Per-unroll summed log-prob of ``ids_ans`` following ``prompt``."""
    import torch

    from traj_geom.extraction.hook import _replicate_coda_head

    ids_prompt = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    ans = torch.tensor([ids_ans], device=model.device, dtype=ids_prompt.dtype)
    ids = torch.cat([ids_prompt, ans], dim=1)
    n_prompt = ids_prompt.shape[1]

    freqs_cis = model.freqs_cis[:, : ids.shape[1]]
    out: list[float] = []
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()

    def hook(_m, _i, o):
        with torch.no_grad():
            logits = _replicate_coda_head(model, o.detach(), freqs_cis).float()
            # position (n_prompt - 1 + k) predicts answer token k
            lp = torch.log_softmax(logits[0], dim=-1)
            total = sum(
                lp[n_prompt - 1 + k, t].item() for k, t in enumerate(ids_ans)
            )
            out.append(total)

    h = mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids, num_steps=max_r)
    finally:
        h.remove()

    arr = np.asarray(out, dtype=np.float64)
    return arr[:max_r] if len(arr) >= max_r else np.pad(
        arr, (0, max_r - len(arr)), constant_values=np.nan
    )


def threshold_depth(margin: np.ndarray, frac: float = 0.9) -> float:
    """Smallest r at which the margin first reaches ``frac`` of its final level.

    Defined on the MARGIN, not on log P(gold), so that a model growing
    uniformly more confident does not register as "solving at depth r".

    The final level is the median of the last quarter of the curve rather than
    its last value: a single unroll's logits are one bf16 sample, and
    docs/rigor_audit.md section 1 showed how much noise that carries.

    Args:
        margin: Per-unroll margin curve.
        frac: Fraction of the final level defining "solved".

    Returns:
        1-indexed depth at which the margin first reaches the target and stays
        at or above it, or NaN if the margin never becomes positive (the model
        never prefers the gold answer, so no threshold exists).
    """
    m = np.asarray(margin, dtype=np.float64)
    if len(m) < 4 or not np.isfinite(m).any():
        return float("nan")
    final = float(np.nanmedian(m[-max(1, len(m) // 4):]))
    if not np.isfinite(final) or final <= 0:
        return float("nan")
    target = frac * final
    ok = m >= target
    # require it to STAY above: a single lucky unroll is not a threshold
    for i in range(len(m)):
        if ok[i] and ok[i:].all():
            return float(i + 1)
    return float("nan")
