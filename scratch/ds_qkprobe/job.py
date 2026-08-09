"""DataSphere Jobs entry point for the QK-alignment probe -- POWERED RUN + CONTROL.

WHY THIS EXISTS. `project_plan.md` G2 names "winding-vs-depth plus a query-key
alignment probe" and records the probe as **0% done** -- the one
supervisor-nominated statistic this project had never touched. Every H2 instrument
built so far (winding D28, effective dimensionality D74(6), the Jacobian argument
D83) reads the HIDDEN STATE. None reads the attention mechanism itself.

WHAT THE SMOKE RUN LEFT OPEN, AND WHY THIS RUN EXISTS.
`bt164frpr04jvqguht5j` passed its self-check (exact 0.0) and banked 24/24 forwards.
Its printed final-unroll means showed `track > local` in all 12 (n_ops, seed) pairs.
That is NOT a result, for two reasons this run fixes:

  1. The per-unroll data never came back -- `config.yaml` declared no `outputs:`,
     so only stdout survived. Fixed: the results JSON is written to the path
     DataSphere passes as argv[1] and declared as an output.
  2. **`track` and `local` prompts END IN DIFFERENT WORDS.** D87/D88 established
     that this model's trajectory geometry reads the INPUT TOKEN, not the
     computation the token selects -- the shape separated two markers at 96.9-100%
     whether or not they selected the same computation. A track-vs-local QK
     difference is therefore fully explained by token identity unless that is
     controlled for. **The smoke run could not distinguish "QK alignment tracks
     reasoning" from "QK alignment reads the last token", which is D88 again.**

THE CONTROL, LIFTED VERBATIM FROM `geometry-marker` (D88), IS THE POINT OF THIS RUN.
Three markers, with the rule block defining **A and C to do exactly the same thing**
and B something different. Every pair differs by exactly ONE character at an
identical token count:

    A vs B  -- different marker, DIFFERENT computation
    A vs C  -- different marker, SAME computation

  * |QK(A) - QK(B)| large AND |QK(A) - QK(C)| ~ 0  -> QK alignment tracks the
    COMPUTATION. This would be the first instrument in the project to do so, and
    would separate the attention mechanism from the hidden-state geometry that
    D88 showed is token-driven.
  * both differences comparable -> QK alignment reads the TOKEN, D88 again, and
    the track/local pattern is input sensitivity wearing a reasoning label.
  * neither differs -> the probe returns nothing on this design, like every other
    H2 instrument, and says so.

**The prediction is written here before the run, per CLAUDE.md section 1: given
D87/D88 and given that every H2 instrument to date returns nothing, the expected
outcome is the SECOND branch -- comparable differences, i.e. token sensitivity.
A clean first branch would be a genuine surprise and the strongest positive result
this project has produced.**

SELF-CHECK BEFORE THE REAL RUN, AND WHY IT IS NOT DECORATION. `verify_extraction`
monkeypatches `scaled_dot_product_attention` for exactly one forward to capture its
GROUND-TRUTH (q, k) arguments and asserts this file's own reconstruction matches.
The FIRST submission of this probe FAILED that check with an error of ~15 -- the
spy was capturing the first attention call in the forward, which belongs to the
first PRELUDE block, not `core_block[-1]`, so it was comparing two unrelated
layers' weights. The run halted before producing a single experimental number.
That is the CLAUDE.md section 5 discipline paying for itself; the check stays.
"""

import json
import os
import subprocess
import sys


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 48
N_OPS = (4, 8, 16, 24)
N_SEEDS = 10          # 4 x 10 x 2 = 80 forwards for the paired arm (smoke had 3)
N_MARKER_ITEMS = 12   # 12 x 3 markers = 36 forwards for the control arm

RULES = ("Rules: for task A, {a}. For task B, {b}. For task C, {a}.\n"
         "Sequence: {seq}\n"
         "Task: {marker}")


def make_variants(n_ops, seed=0):
    """Verbatim from `src/traj_geom/shapes/synthetic.py`, pinned by
    `tests/test_kernel_tasks.py` project-wide: `track` and `local` share a
    byte-identical body and differ only in the trailing question."""
    import random
    rng = random.Random(seed)
    ops = [rng.choice([1, -1]) for _ in range(n_ops)]
    body = "Start at 0. " + " ".join("Add 1." if o > 0 else "Subtract 1." for o in ops)
    return {
        "track": (body + " Final total? A:", str(sum(ops))),
        "local": (body + " What was the last instruction? A:",
                  "Add" if ops[-1] > 0 else "Subtract"),
    }


def marker_items(n=N_MARKER_ITEMS):
    """Verbatim design from `scratch/kaggle_marker/body.py` (D88): A and C select
    the SAME computation, B a different one, and every prompt differs from every
    other by exactly one character at an identical token count. Golds are single
    characters so rank never depends on tokenisation (the D89 defect)."""
    import random
    out = []
    for s in range(n):
        rng = random.Random(s * 7919 + 104729)
        bits = [rng.randint(0, 1) for _ in range(8)]
        base = {"a": "report how many 1s are in the sequence",
                "b": "report the last symbol of the sequence",
                "seq": " ".join(map(str, bits))}
        g = {"A": str(sum(bits)), "B": str(bits[-1]), "C": str(sum(bits))}
        for m in ("A", "B", "C"):
            out.append({"item": s, "marker": m,
                        "prompt": RULES.format(marker=m, **base), "gold": g[m]})
    return out


def build_qk_extractor(model):
    """Returns (attn, reconstruct_qk) for per-unroll QK alignment at the answer
    position, on `model.transformer.core_block[-1].attn`."""
    attn = model.transformer.core_block[-1].attn
    mod_name = type(attn).__module__
    rotary_fn = sys.modules[mod_name].apply_rotary_emb_complex_like
    n_head, n_kv_heads, head_dim = attn.n_head, attn.n_kv_heads, attn.head_dim
    chunks = attn.chunks
    has_qk_bias = bool(getattr(attn.config, "qk_bias", False))

    def reconstruct_qk(raw_qkv, freqs_cis):
        """Copied verbatim from `CausalSelfAttention.forward` up to and including
        the rotary call. The ROTATION is the model's OWN function, never
        re-derived -- reimplementing rotary math by hand is the D71-shaped
        mistake this project has already made once."""
        b, s, _ = raw_qkv.shape
        q, k, _v = raw_qkv.split(chunks, dim=2)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_kv_heads, head_dim)
        if has_qk_bias:
            q_bias, k_bias = attn.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        q, k = rotary_fn(q, k, freqs_cis=freqs_cis)
        return q, k

    return attn, reconstruct_qk


def verify_extraction(model, tok, torch):
    """THE SELF-CHECK. Ground truth is the (q, k) that `scaled_dot_product_attention`
    is ACTUALLY called with. If this disagrees with our reconstruction, the run
    stops rather than producing numbers nobody checked."""
    import torch.nn.functional as F

    attn, reconstruct_qk = build_qk_extractor(model)
    ground_truth = {}
    raw_qkv_seen = {}
    capturing = {"on": False}

    # A single forward calls scaled_dot_product_attention many times: 2 prelude
    # blocks, then the core-block unrolls, then 2 coda blocks. The FIRST call
    # belongs to the first PRELUDE block, NOT core_block[-1] -- a naive "capture
    # the first call" spy compares two unrelated layers and fails with an error
    # of ~15 (this happened, job bt1jfkkdf71ms695e6id). Bracket the capture with
    # pre/post hooks on `attn` ITSELF so it arms only while core_block[-1]'s own
    # forward is on the call stack.
    h_wqkv = attn.Wqkv.register_forward_hook(
        lambda _m, _i, o: raw_qkv_seen.__setitem__("v", o.detach()))
    h_pre = attn.register_forward_pre_hook(
        lambda _m, _i: capturing.__setitem__("on", True))
    h_post = attn.register_forward_hook(
        lambda _m, _i, _o: capturing.__setitem__("on", False))

    orig_sdpa = F.scaled_dot_product_attention

    def spy_sdpa(q, k, v, *a, **kw):
        if capturing["on"] and "q" not in ground_truth:
            ground_truth["q"] = q.detach().clone()
            ground_truth["k"] = k.detach().clone()
        return orig_sdpa(q, k, v, *a, **kw)

    ids = tok("Verify.", return_tensors="pt").input_ids.to(model.device)
    freqs = model.freqs_cis[:, : ids.shape[1]]
    F.scaled_dot_product_attention = spy_sdpa
    try:
        with torch.no_grad():
            model(input_ids=ids, num_steps=1)
    finally:
        F.scaled_dot_product_attention = orig_sdpa
        h_wqkv.remove()
        h_pre.remove()
        h_post.remove()

    q_hat, k_hat = reconstruct_qk(raw_qkv_seen["v"], freqs)
    q_err = (q_hat.transpose(1, 2) - ground_truth["q"]).abs().max().item()
    k_err = (k_hat.transpose(1, 2) - ground_truth["k"]).abs().max().item()
    print(f"  self-check: max|q_hat - q_true| = {q_err:.3e}, "
          f"max|k_hat - k_true| = {k_err:.3e}", flush=True)
    return q_err, k_err


def main():
    # DataSphere passes the declared output path as argv[1]. Capture the original
    # working directory BEFORE chdir'ing into the clone, or the results land
    # somewhere the platform will not collect -- which is exactly how the smoke
    # run lost its per-unroll data.
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("qk_probe.json")
    print(f"results will be written to {out_path}", flush=True)

    print("Cloning the repository...", flush=True)
    run(
        "git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo"
    )
    os.chdir("repo")

    print("Pinning torch to the driver-compatible, flex_attention-capable build "
          "the project's own gpu-smoke/blayney-repro jobs established.", flush=True)
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("Verifying accelerator...", flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
          else "none", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION,
                                     trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    print("\n=== SELF-CHECK: does the hook-based (q, k) match ground truth? ===",
          flush=True)
    q_err, k_err = verify_extraction(model, tok, torch)
    payload = {"self_check": {"q_err": q_err, "k_err": k_err}}
    if q_err > 1e-4 or k_err > 1e-4:
        print("SELF-CHECK FAILED -- stopping before any experimental number is "
              "produced. Nothing below this line may be trusted.", flush=True)
        payload["self_check"]["passed"] = False
        with open(out_path, "w") as fh:
            json.dump(payload, fh)
        return
    payload["self_check"]["passed"] = True
    print("SELF-CHECK PASSED.\n", flush=True)

    attn, reconstruct_qk = build_qk_extractor(model)

    def coda_head(h_state, freqs_cis):
        """Verbatim D71-validated tail, for the rank readout alongside QK."""
        x = model.transformer.ln_f(h_state)
        block_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        for block in model.transformer.coda:
            block_idx -= 1
            x = block(x, freqs_cis, block_idx, None, None)
        x = model.transformer.ln_f(x)
        return model.lm_head(x)

    def qk_forward(prompt, gold):
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        g_ids = tok(gold, add_special_tokens=False).input_ids
        n_p = ids.shape[1]
        freqs = model.freqs_cis[:, :n_p]

        raw_seen = {}
        h_wqkv = attn.Wqkv.register_forward_hook(
            lambda _m, _i, o: raw_seen.__setitem__("v", o.detach()))

        cos_per_unroll, ranks = [], []
        mod = model.transformer.core_block[-1]
        mod._forward_hooks.clear()

        def hook(_m, _i, o):
            with torch.no_grad():
                st = o.detach()
                q, k = reconstruct_qk(raw_seen["v"], freqs)
                qa, ka = q[0, n_p - 1], k[0, n_p - 1]     # answer position, all heads
                cos = torch.nn.functional.cosine_similarity(qa, ka, dim=-1)
                cos_per_unroll.append(cos.float().cpu().numpy().tolist())
                row = torch.log_softmax(
                    coda_head(st, freqs).float()[0, n_p - 1], dim=-1)
                ranks.append(int((row > row[g_ids[0]]).sum().item()) + 1)

        h = mod.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
            h_wqkv.remove()
        return {"cos_per_unroll": cos_per_unroll, "rank_curve": ranks,
                "n_tokens": int(n_p), "best_rank": int(min(ranks)),
                "multi_token_gold": bool(len(g_ids) > 1)}

    def bank(records, label, prompt, gold, meta):
        try:
            r = qk_forward(prompt, gold)
            r.update(meta)
            r["ok"] = True
            mc = sum(r["cos_per_unroll"][-1]) / len(r["cos_per_unroll"][-1])
            print(f"  {label}: final-unroll mean cos(q,k) = {mc:+.4f}  "
                  f"tokens={r['n_tokens']} best_rank={r['best_rank']}", flush=True)
        except Exception as exc:
            r = {**meta, "ok": False, "why": f"{type(exc).__name__}: {exc}"}
            print(f"  {label}: FAILED {r['why']}", flush=True)
        records.append(r)

    print("=== ARM 1: paired track/local, matched n_ops (D83 design) ===",
          flush=True)
    paired = []
    for n_ops in N_OPS:
        for seed in range(N_SEEDS):
            for kind, (prompt, gold) in make_variants(n_ops, seed=seed).items():
                bank(paired, f"n_ops={n_ops:>2} seed={seed} {kind:>6}", prompt, gold,
                     {"kind": kind, "n_ops": n_ops, "seed": seed})

    print("\n=== ARM 2: marker control, A/B/C (D88 design) -- "
          "A and C are the SAME computation, B different, all one char apart ===",
          flush=True)
    control = []
    for it in marker_items():
        bank(control, f"item={it['item']:>2} marker={it['marker']}",
             it["prompt"], it["gold"],
             {"item": it["item"], "marker": it["marker"], "gold": it["gold"]})

    n_ok = sum(1 for r in paired + control if r.get("ok"))
    print(f"\n=== banked {n_ok}/{len(paired) + len(control)} ===", flush=True)

    payload["paired"] = paired
    payload["control"] = control
    payload["config"] = {"num_steps": NUM_STEPS, "n_ops": list(N_OPS),
                         "n_seeds": N_SEEDS, "n_marker_items": N_MARKER_ITEMS,
                         "model": MODEL_ID, "revision": REVISION}
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {out_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
