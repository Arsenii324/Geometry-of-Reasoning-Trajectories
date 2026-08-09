"""Guards that make this project's recurring failure modes loud instead of silent.

Every function here exists because the failure it prevents ACTUALLY HAPPENED, most
of them on 2026-08-09, several of them twice. Each carries its occasion, because a
guard whose motivating bug is forgotten gets deleted as ceremony.

The design rule (from `numerical-research-code`): a research bug does not throw, it
returns a plausible float. So these RAISE by default and offer an explicit opt-out
that forces the caller to say how the ambiguity was resolved. They are cheap enough
to call unconditionally.

THE FIVE FAILURE MODES, and the guard for each:

1. A TEST THAT CANNOT FIRE.  `require_resolvable_alpha`
   `run_h0_within.py`'s planted-effect floor used 400 permutations for the main
   test but a default of 40 for the floor, against a Bonferroni alpha of 0.00625.
   The smallest p a k-permutation test can return is 1/(k+1); 1/41 = 0.024 can
   never clear 0.00625, so the floor reported 0% detection at EVERY effect size
   including 1.0 sd. That reads as "hopelessly underpowered design" and actually
   means "this check is arithmetically incapable of returning anything".

2. A p-VALUE BELOW ITS OWN EXACT FLOOR.  `require_resolvable_alpha` + `exact_floor`
   `run_qk_analysis.py` reported p = 0.00015 from a 12-pair sign-flip test. That
   test has only 2**12 = 4096 distinct outcomes, so its smallest attainable p is
   1/4096 = 0.000244. The reported value was a Monte-Carlo artefact of
   (hits+1)/(N+1) with N = 20000 sampling a 4096-point space.

3. REPLICATES COUNTED AS INDEPENDENT OBSERVATIONS.  `independent_n`, `require_units`
   Four occurrences in one day. D94 quoted "0 of ~960 orbit pairs" when contraction
   is a property of the PROMPT and n_eff was 16. D96's confound correlation pooled
   24 observations that were 12 items x 2 contrasts sharing a value, giving
   p = 2.4e-05 where an item-clustered permutation gives 0.093. D100 pooled 3 items
   per level undisclosed. D101 caught it and corrected -- which is why D101 is the
   row that survived audit unchanged.

4. A CONVENIENCE SLICE PRESENTED AS A SAMPLE.  `require_units`, `describe_range`
   D99 said "40 independent answer-position orbits, median relative sd 1.30e-04".
   The slice was `sorted(glob)[:40]` = 32 replicates of one item plus 8 of another:
   1.25 ITEMS. Over the true 512-orbit population the median is 5.04e-05, 2.6x
   smaller. D98 did the same thing differently, quoting one prompt's four per-block
   residuals ("0.09-0.11") as an across-prompt range that matched NO prompt.

5. A VERDICT EMITTED OVER NO DATA.  `require_evidence`
   `ds_cyclenull` exited SUCCESS and printed "NO statistic separates trained from
   untrained => D98's cycle is ARCHITECTURAL, not learned" while all three
   untrained arms had died of CUDA OOM and the comparison ran over 0/0 statistics.
   An `if not significant:` branch fired on an EMPTY list. D95's kernel had the
   same shape and was only caught because it printed VOID into a log nobody would
   have read.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any


class RigorError(AssertionError):
    """Raised when a result would be un-interpretable rather than merely wrong.

    Deliberately an AssertionError subclass so it is not caught by `except
    Exception` blocks that wrap per-item work in the kernels.
    """


def exact_floor(n_perm: int) -> float:
    """Smallest p-value a permutation test with `n_perm` draws can report.

    The convention throughout this project is p = (hits + 1) / (n_perm + 1), whose
    minimum at hits = 0 is 1 / (n_perm + 1).
    """
    if n_perm < 1:
        raise RigorError(f"n_perm must be >= 1, got {n_perm}")
    return 1.0 / (n_perm + 1)


def require_resolvable_alpha(n_perm: int, alpha: float, *, what: str = "test") -> None:
    """Raise if `alpha` is unreachable by a permutation test with `n_perm` draws.

    Occasion: failure mode 1. Call this BEFORE running the permutations, not after
    reading a suspicious 0% detection rate.
    """
    floor = exact_floor(n_perm)
    if floor >= alpha:
        raise RigorError(
            f"{what}: alpha={alpha:.6g} is UNREACHABLE with n_perm={n_perm}. "
            f"The smallest attainable p is 1/(n_perm+1)={floor:.6g}. This check "
            f"cannot fire regardless of effect size -- a 0% detection rate would "
            f"mean nothing. Need n_perm > {math.ceil(1 / alpha) - 1}."
        )


def require_exact_test_if_small(n_pairs: int, n_perm: int, *, what: str = "test") -> None:
    """Raise if the permutation SPACE is smaller than the number of draws.

    With `n_pairs` sign-flip pairs there are only 2**n_pairs distinct outcomes.
    Sampling more than that with replacement produces p-values below the exact
    floor, which are artefacts. Enumerate instead.

    Occasion: failure mode 2.
    """
    if n_pairs < 1:
        raise RigorError(f"{what}: n_pairs must be >= 1, got {n_pairs}")
    space = 2 ** n_pairs
    if n_perm > space:
        raise RigorError(
            f"{what}: sampling {n_perm} permutations from a space of only "
            f"2**{n_pairs}={space} produces p-values below the exact floor "
            f"1/{space}={1/space:.6g}. Enumerate the {space} sign patterns exactly."
        )


def independent_n(keys: Sequence[Any]) -> int:
    """Number of INDEPENDENT units behind a sequence of per-observation keys.

    `keys` is one key per observation naming the unit it belongs to (prompt, item,
    weight-set). This is the number that belongs in a claim, not `len(keys)`.

    Occasion: failure mode 3.
    """
    return len(set(keys))


def require_units(
    keys: Sequence[Any], min_units: int, *, what: str = "sample",
    allow_replicates: bool = False,
) -> int:
    """Raise unless the observations span at least `min_units` independent units.

    This catches BOTH pseudoreplication (many rows, few units) and the convenience
    slice (`glob[:40]` that turns out to be one item measured 40 times).

    Set `allow_replicates=True` only when you have said, in the calling code, what
    the independent unit is and why replicates are acceptable here.

    Occasions: failure modes 3 and 4.
    """
    n_units = independent_n(keys)
    if not allow_replicates and n_units < min_units:
        from collections import Counter
        top = Counter(keys).most_common(3)
        raise RigorError(
            f"{what}: {len(keys)} observations span only {n_units} independent "
            f"unit(s), fewer than the {min_units} required. Most frequent: {top}. "
            f"Quoting n={len(keys)} here would be pseudoreplication; the honest n "
            f"is {n_units}. If replicates are intended, pass allow_replicates=True "
            f"and state the independent unit in the caller."
        )
    return n_units


def require_evidence(n_usable: int, *, what: str, minimum: int = 1) -> None:
    """Raise before emitting a verdict if there is nothing to conclude from.

    Put this at the TOP of any branch that prints or returns a conclusion. The
    failure it prevents is not a wrong verdict but a confident one over an empty
    result set, which no reader can distinguish from a real null.

    Occasion: failure mode 5.
    """
    if n_usable < minimum:
        raise RigorError(
            f"{what}: refusing to emit a verdict from {n_usable} usable "
            f"measurement(s) (minimum {minimum}). This is VOID, not NULL -- the "
            f"measurement did not happen, which is a different finding from the "
            f"effect being absent. Report it as VOID and say the question is open."
        )


def describe_range(
    values: Iterable[float], keys: Sequence[Any], *, what: str = "range",
) -> str:
    """Format a min-max range together with the number of independent units.

    Ranges in this project have twice been one unit's spread presented as if it
    spanned the data. Making the unit count part of the STRING means a range
    cannot be quoted without it.

    Occasion: failure mode 4.
    """
    vals = list(values)
    if not vals:
        raise RigorError(f"{what}: empty values")
    if len(vals) != len(keys):
        raise RigorError(
            f"{what}: {len(vals)} values but {len(keys)} keys -- they must be "
            f"per-observation and aligned."
        )
    n_units = independent_n(keys)
    lo, hi = min(vals), max(vals)
    med = sorted(vals)[len(vals) // 2]
    return (f"{what}: {lo:.4g}-{hi:.4g} (median {med:.4g}) over {len(vals)} "
            f"observations spanning {n_units} independent unit(s)")
