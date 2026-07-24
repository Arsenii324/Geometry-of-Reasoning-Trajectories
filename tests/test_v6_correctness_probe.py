"""Regression-pins the verified top-k D12 result (claims_ledger.md D12,
2026-07-24 update) against the real cached results/v6_correctness_probe.csv.
"""

from __future__ import annotations

import pandas as pd


def test_v6_correctness_probe_reproduces_verified_d12_topk_result() -> None:
    """13/24 rows are honestly-verifiable single-token answers; on those,
    top-5 hit rate is 13/13 (100%), strict argmax only 4/13.
    """
    df = pd.read_csv("results/v6_correctness_probe.csv")
    assert len(df) == 24

    single = df[df["is_single_token_answer"]]
    multi = df[~df["is_single_token_answer"]]
    assert len(single) == 13
    assert len(multi) == 11

    topk_hits = int((single["topk_correct_at_step"] >= 0).sum())
    argmax_hits = int((single["correct_at_step"] >= 0).sum())
    assert topk_hits == 13
    assert argmax_hits == 4

    # Every strict-argmax hit on the single-token subset is target=0 --
    # the small-number-prior signature the original D12 reading was built on.
    argmax_hit_targets = single.loc[single["correct_at_step"] >= 0, "target"]
    assert (argmax_hit_targets == 0).all()

    # Multi-token targets are exactly the negative ones in this task
    # (counting can go negative; the >=10 case never arises at these depths).
    assert (multi["target"] < 0).all()
    assert (multi["n_answer_tokens"] == 2).all()
