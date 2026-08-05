"""Shared kernel blocks, inlined into `main.py` by `scripts/build_kernel.py`.

WHY THIS EXISTS
    Kaggle kernels are single self-contained files, so every bundle re-declared
    the same helpers. Measured across the 23 bundles in `scratch/`: 4563 lines,
    28% of all 8-line windows appearing in more than one kernel, with `fit_rho`,
    `orbit`, `measure` and `summarise` written 8-9 times each.

    That is not merely wasteful. `np.arange(1, n+1, float)` -- which passes
    `float` as the STEP argument instead of the dtype -- was fixed once and then
    reintroduced by copy-paste twice more, costing one failed GPU run. Duplicated
    code re-imports fixed bugs.

    Blocks here are unit-tested locally (`tests/test_kernel_common.py`) against
    known answers, so a kernel built from them starts from tested code. Anything
    genuinely one-off still belongs in the kernel body.

USAGE
    A kernel body declares what it needs on one line:

        # @needs: fit_rho orbit measure summarise prompts_by_family

    and `python -m scripts.build_kernel <bundle>` emits `main.py` = body docstring
    + those blocks (with their dependencies) + the body. Blocks are emitted in
    dependency order and never partially.
"""

# ---8<--- run
def run(cmd):
    """Shell out, echoing the command so the Kaggle log shows what was installed."""
    import subprocess
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)
# ---8<---


# ---8<--- fit_rho
def fit_rho(curve, k=3.0, tail_frac=0.25):
    """Contraction rate from a decaying curve -> (rho, n_used, fit_r2).

    Fits ``log(curve)`` linear in t over the PRE-FLOOR regime only. Finite
    arithmetic floors any such curve; including floored points drags the slope
    toward zero and INFLATES rho, which is the single most common way a
    contraction estimate goes wrong here. Points are kept only while above
    ``k * floor``, floor = median of the last ``tail_frac`` of the run.

    Returns nan (not a number) when fewer than 4 points survive -- a fit on three
    points is not a measurement and must not be silently reported as one.
    """
    import numpy as np
    y = np.asarray(curve, dtype=np.float64)
    if len(y) < 6:
        return float("nan"), 0, float("nan")
    floor = float(np.median(y[-max(3, int(len(y) * tail_frac)):]))
    if not np.isfinite(floor) or floor <= 0:
        return float("nan"), 0, float("nan")
    n = 0
    for v in (y > k * floor):        # leading contiguous run; once floored it stays
        if not v:
            break
        n += 1
    if n < 4:
        return float("nan"), int(n), float("nan")
    t = np.arange(n, dtype=np.float64)
    ly = np.log(y[:n])
    slope, icpt = np.polyfit(t, ly, 1)
    pred = slope * t + icpt
    denom = ((ly - ly.mean()) ** 2).sum()
    r2 = float(1 - ((ly - pred) ** 2).sum() / denom) if denom > 0 else float("nan")
    return float(np.exp(slope)), int(n), r2
# ---8<---


# ---8<--- capture_unrolls
def capture_unrolls(model, ids, seed, max_r, last_pos_only=True):
    """Core-block output at each unroll -> [max_r, hidden] (or [max_r, pos, hidden]).

    `torch.manual_seed(seed)` controls Huginn's random initial latent, which is
    what makes two-orbit convergence possible. Hooks are cleared first and removed
    in a finally, so a raised exception cannot leave a hook attached to the model.
    """
    import numpy as np
    import torch
    cap = []
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()
    sel = (lambda o: o.detach()[0, -1, :]) if last_pos_only else (lambda o: o.detach()[0])
    h = mod.register_forward_hook(
        lambda m, i, o, c=cap: c.append(sel(o).float().cpu().numpy()))
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            model(input_ids=ids, num_steps=max_r)
    finally:
        h.remove()
    return np.stack(cap)
# ---8<---


# ---8<--- measure_rho  needs: fit_rho capture_unrolls
def measure_rho(model, tok, label, prompts, max_r=128):
    """Two-orbit convergence AND step-norm decay per prompt.

    Two estimators because each has a known failure mode: two-orbit measures the
    contraction of the MAP independent of where the fixed point sits, while
    step-norm is cheaper but contaminated if the orbit has not reached the linear
    regime. Agreement is the internal check; disagreement is reportable.

    `prompts` is a list of (family, text). `d0` is recorded so a reader can verify
    the two orbits actually started apart -- if they did not, the whole
    measurement is vacuous.
    """
    import numpy as np
    import torch
    rows = []
    for i, (family, p) in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        a = capture_unrolls(model, ids, 1000 + i, max_r)
        b = capture_unrolls(model, ids, 2000 + i, max_r)
        d = np.linalg.norm(a - b, axis=1)
        s = np.linalg.norm(np.diff(a, axis=0), axis=1)
        rd, nd, fd = fit_rho(d)
        rs, ns, fs = fit_rho(s)
        rows.append({"prompt": i, "family": family, "n_tok": int(ids.shape[1]),
                     "rho_orbit": rd, "n_orbit": nd, "r2_orbit": fd,
                     "rho_step": rs, "n_step": ns, "r2_step": fs,
                     "d0": float(d[0]), "d_end": float(d[-1]),
                     "norm_h": float(np.linalg.norm(a[-1]))})
        print(f"    {label} {family[:5]}{i}: rho_orbit={rd:.4f}(n={nd},fit {fd:.3f})  "
              f"rho_step={rs:.4f}(n={ns},fit {fs:.3f})  d0={d[0]:.2f}  "
              f"||h||={rows[-1]['norm_h']:.2f}", flush=True)
        del ids
        torch.cuda.empty_cache()
    return rows
# ---8<---


# ---8<--- summarise
def summarise(rows, keys=("rho_orbit", "rho_step"), clean_key="r2_orbit", bar=0.9):
    """Per-arm summary, reporting BOTH all-fit and clean-fit means.

    Reporting only one hides an analyst choice that has already flipped a
    conclusion in this project (claims_ledger D52(2)): dropping fits below the
    R^2 bar moved a within-training trend from p=0.036 to p=0.19. Both are
    emitted so the sensitivity is visible without a rerun.
    """
    import numpy as np

    def agg(rs, key):
        v = np.array([r[key] for r in rs], float)
        v = v[np.isfinite(v)]
        return {"mean": float(v.mean()) if len(v) else float("nan"),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                "n": int(len(v))}

    good = [r for r in rows if r.get(clean_key, 1.0) > bar]
    out = {k: agg(rows, k) for k in keys}
    out["clean"] = {k: agg(good, k) for k in keys} if good else {}
    out["n_below_bar"] = len(rows) - len(good)
    out["by_family"] = {}
    for fam in sorted({r.get("family", "?") for r in rows}):
        rs = [r for r in rows if r.get("family", "?") == fam]
        out["by_family"][fam] = {k: agg(rs, k) for k in keys}
    return out
# ---8<---


# ---8<--- prompts_by_family
def prompts_by_family(n_per=3, m=64):
    """Four task families with token lengths spanning ~15 to ~74.

    Stratifying by family is what lets a per-model quantity be tested for
    constancy across tasks instead of assumed (claims_ledger D45). Length varies
    WITH family here by design, so the two are collinear -- a real limitation,
    recorded rather than hidden.
    """
    import random
    out = []
    for i in range(n_per):
        rng = random.Random(i * 7919)
        bits = [1 if rng.random() < 0.5 else 0 for _ in range(m)]
        out.append(("counting",
                    "Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:"))
    for i in range(n_per):
        rng = random.Random(i * 104729)
        seq, d = [], 0
        for _ in range(m // 2):
            if d == 0 or (rng.random() < 0.5 and d < 8):
                seq.append("(")
                d += 1
            else:
                seq.append(")")
                d -= 1
        seq += [")"] * d
        out.append(("nesting",
                    "String: " + " ".join(seq) + ". What is the maximum nesting depth? A:"))
    for i in range(n_per):
        rng = random.Random(i * 15485863)
        a, b, c = rng.randint(11, 99), rng.randint(3, 19), rng.randint(2, 9)
        out.append(("arith",
                    f"A shop had {a} boxes. It sold {b} boxes each day for {c} days. "
                    f"How many boxes are left? A:"))
    stems = ["The man picked up the heavy suitcase and walked toward the platform. He",
             "She opened the oven, checked the bread, and decided it needed more time. Then she",
             "The dog heard the doorbell, ran into the hallway, and started barking. Next it"]
    for i in range(n_per):
        out.append(("commonsense", stems[i % len(stems)]))
    return out
# ---8<---


# ---8<--- load_arm
def load_arm(spec, cfg, revision=None):
    """Load one weight-set. `spec` is None for a fresh random init, else a repo id.

    Backfills config attributes ABSENT from an older checkpoint from the final
    model's config -- intermediate Huginn checkpoints predate fields the current
    modeling code reads (`test_time_noise`), and without this they raise
    AttributeError. Only missing keys are copied, and every backfill is logged.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    if spec is None:
        torch.manual_seed(revision if isinstance(revision, int) else 0)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    else:
        ck = AutoConfig.from_pretrained(spec, revision=revision, trust_remote_code=True)
        added = [k for k, v in vars(cfg).items()
                 if not hasattr(ck, k) and not k.startswith("_")]
        for k in added:
            setattr(ck, k, getattr(cfg, k))
        if added:
            print(f"  backfilled {len(added)} config attrs: {sorted(added)}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec, revision=revision, config=ck, trust_remote_code=True,
            low_cpu_mem_usage=True)
    return model.to(torch.float32).to("cuda").eval()
# ---8<---


# ---8<--- free_arm
def free_arm(model, repo_id=None):
    """Release a weight-set and report what was actually reclaimed.

    Printing free memory is the point: a silent cleanup is how eight checkpoints
    were lost to OOM with 13.46 GiB still held (see `geometry-rho-direct`). Also
    purges the HF cache for `repo_id`, since ten 7GB checkpoints exhaust the disk.
    """
    import gc
    import os
    import shutil

    import torch
    try:
        model = model.to("cpu") if model is not None else None
    except Exception:                                          # noqa: BLE001, S110
        pass
    del model
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"  after cleanup: {free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB",
          flush=True)
    if repo_id:
        d = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                         "models--" + repo_id.replace("/", "--"))
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print(f"  purged {d}", flush=True)
# ---8<---


# ---8<--- cv_r2
def cv_r2(x, y, groups=None, alpha=1e3, n_splits=5, n_null=0):
    """Held-out R^2 with an optional permutation null -> (r2, null_list).

    `groups` switches to GroupKFold, which is REQUIRED whenever rows come from the
    same prompt: a plain KFold leaks between positions of one sequence and inflates
    the score. The permutation null is the check that d >> n has not turned the
    probe into an interpolator -- an honest held-out null lands BELOW zero.
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mdl = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    cv = (GroupKFold(n_splits=n_splits) if groups is not None
          else KFold(n_splits=n_splits, shuffle=True, random_state=0))

    def score(target):
        p = cross_val_predict(mdl, x, target, cv=cv, groups=groups)
        return float(1 - ((target - p) ** 2).sum() / ((target - target.mean()) ** 2).sum())

    r2 = score(y)
    rng = np.random.default_rng(0)
    null = [score(rng.permutation(y)) for _ in range(n_null)]
    return r2, null
# ---8<---


# ---8<--- batched_generate
def batched_generate(model, tok, prompts, max_new=24, num_steps=32, verbose=True,
                     continuous_compute=False, max_batch_tokens=1024):
    """Greedy generation over length-homogeneous batches, WITHOUT the KV cache.

    WHY NOT THE MODEL'S OWN `generate_minimal`. It is batched and cache-backed and
    should be strictly better. It returned an EMPTY STRING for every prompt in
    `geometry-task-accuracy` (D62) while the naive batch-1 loop reached 100% on the
    same task and prompts (D60). Two candidate mechanisms were checked against the
    source and BOTH REFUTED: it returns a plain tensor unless `return_dict_in_generate`
    is set (it was not), and the stop check reads `next_token[i,0]` only, so the
    prompt's own `<|begin_text|>` cannot trip it. What remains untested is the
    cache+batch path itself. Rather than debug someone else's decode loop on a
    borrowed GPU, this keeps the generator that is KNOWN to work and takes the
    speedup from batching alone.

    Batching still pays: the screen ran 240 completions at ~1 min each because it
    was batch-1 (C9). Bucket sizes here are 8-16, so most of the win survives.

    ONE ARCHITECTURE-SPECIFIC TRAP, verified in the source: `forward` sets
    `prepared_attn_mask = None` -- the attention mask is commented out -- so PADDING
    IS NOT MASKED and a padded batch silently attends to pad tokens. Batching is
    therefore only safe across prompts of IDENTICAL token length, and lengths are
    measured rather than assumed (ten six-letter words through one template tokenise
    to 15 OR 16 tokens).

    `continuous_compute` warm-starts each new token's latent from the previous
    token's final latent instead of re-initialising it randomly -- Huginn's
    "continuous CoT" mode, which no kernel in this project had ever used. It needs
    `output_details` to return latents, so it is requested explicitly.

    `max_batch_tokens` caps `batch x sequence_length`, because this loop has no KV
    cache and therefore re-runs the FULL growing sequence every step. float32
    weights are ~14.1 GB of a T4's 14.56 GB, leaving ~450 MB for activations, and
    the gated MLP's inner width is 17920 -- so batch 16 x ~100 tokens OOM'd inside
    `nonlin(x_fc_1) * x_fc_2`. Buckets are split into chunks satisfying
    `chunk * (prompt_len + max_new) <= max_batch_tokens`, which keeps the peak
    bounded regardless of how long the prompts or completions are.

    Raises if every output is empty: that is the D62 symptom, and returning it
    silently is what let a full table of zeros read as a capability finding.
    """
    import torch

    dev = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    stop = {65504, 65505, 65508}                      # begin_text, end_text, end_turn
    if getattr(tok, "eos_token_id", None) is not None:
        stop.add(tok.eos_token_id)

    enc = [tok(p, return_tensors="pt", add_special_tokens=False).input_ids[0] for p in prompts]
    buckets: dict[int, list[int]] = {}
    for i, e in enumerate(enc):
        buckets.setdefault(int(e.shape[0]), []).append(i)
    # split each length-bucket so batch x seq stays inside the activation budget
    chunks: list[list[int]] = []
    for width, idxs in buckets.items():
        per = max(1, max_batch_tokens // max(width + max_new, 1))
        chunks += [idxs[k:k + per] for k in range(0, len(idxs), per)]
    if verbose:
        print(f"    {len(prompts)} prompts -> {len(buckets)} length-buckets -> "
              f"{len(chunks)} chunks (max {max(len(c) for c in chunks)} per chunk, "
              f"budget {max_batch_tokens} tok)", flush=True)

    out: list[str] = [""] * len(prompts)
    for idxs in chunks:
        ids = torch.stack([enc[i] for i in idxs]).to(dev)
        n_prompt = ids.shape[1]
        live = [True] * len(idxs)
        state = None
        for _ in range(max_new):
            kw = {"num_steps": num_steps}
            if continuous_compute:
                kw["output_details"] = {"return_logits": True, "return_latents": True,
                                        "return_head": False, "return_stats": False}
                if state is not None:
                    kw["input_states"] = state
            with torch.no_grad():
                res = model(input_ids=ids, **kw)
            logits = res.logits if hasattr(res, "logits") else res[0]
            if continuous_compute:
                lat = getattr(res, "latent_states", None)
                state = lat[:, -1:, :].clone() if lat is not None else None
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            for b in range(len(idxs)):
                if int(nxt[b, 0]) in stop:
                    live[b] = False
            ids = torch.cat([ids, nxt], dim=1)
            if not any(live):
                break
        for b, i in enumerate(idxs):
            out[i] = tok.decode(ids[b, n_prompt:], skip_special_tokens=True)
        del ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not any(o.strip() for o in out):
        raise RuntimeError(
            f"batched_generate produced empty output for all {len(prompts)} prompts. "
            "This is the D62 failure mode; refusing to return silently.")
    return out
# ---8<---


# ---8<--- assert_generation_works  needs: batched_generate
def assert_generation_works(model, tok, chat=True):
    """Smoke-test the generator against a task the model provably does, before use.

    D62: `batched_generate` returned an empty string for every prompt and the whole
    task-accuracy run was scored on it. Its unit tests passed -- they asserted the
    length-bucketing the author designed and never that the function returns correct
    text. A block that talks to a real model needs a real check against that model.

    `copy` is used because D60 measured it at 100% under the chat template, so a
    failure here is unambiguous. Raises rather than warns: a silent generator is
    exactly what produced a full run of zeros that read as a scientific result.
    """
    words = ["banana", "orange", "puzzle", "kitten"]
    bodies = [f"Repeat this word exactly.\nWord: {w}" for w in words]
    if chat:
        texts = [tok.apply_chat_template([{"role": "user", "content": b}],
                                         tokenize=False, add_generation_prompt=True)
                 for b in bodies]
    else:
        texts = [b + "\nAnswer:" for b in bodies]
    got = batched_generate(model, tok, texts, max_new=8, verbose=False)
    hits = sum(w in g.lower() for w, g in zip(words, got, strict=True))
    print(f"  generation smoke-test: {hits}/{len(words)} copied  -> {got}", flush=True)
    if hits < len(words) // 2:
        raise RuntimeError(
            f"generation smoke-test FAILED ({hits}/{len(words)}); got {got}. "
            "Refusing to run an experiment on a generator that cannot copy a word.")
    return hits
# ---8<---


# ---8<--- cv_r2_nonlinear
def cv_r2_nonlinear(x, y, groups=None, n_splits=5, n_null=0, seed=0, n_pc=8):
    """Held-out R^2 from a NONLINEAR probe, as a companion to the linear one.

    WHY THIS EXISTS. Every content measurement in this project is a ridge probe,
    i.e. strictly linear (D41, D48, D50, D53). That licenses exactly one reading of
    the headline -- "training does not change what the state contains" -- and
    forbids another no measurement here can distinguish: **training may encode the
    same content NONLINEARLY.** If the trained model represents the quantity in a
    curved way while random weights preserve it linearly (which prompt re-injection
    would do), a linear probe favours the UNTRAINED arm for reasons unrelated to
    information content. That is exactly the D41/D48 pattern, so it is a live
    alternative explanation rather than a hypothetical.

    PCA to `n_pc`=8 components, then degree-2 polynomial ridge. `n_pc` is the
    load-bearing setting and was tuned against synthetics, not guessed: at 20 PCs
    the expansion is 230 features from ~64 training rows and R^2 caps at 0.36 even
    on a CLEAN LINEAR target; at 8 PCs it is 44 features and reaches 0.725 linear /
    0.620 quadratic with a null of -0.268. Regularisation strength barely matters
    (0.354-0.356 across alpha 0.1-10); the feature-count ratio is everything.

    THREE earlier designs were rejected against synthetic targets with KNOWN
    structure, because a weak nonlinear probe biases the test toward confirming the
    headline it exists to challenge:
      * MLP(64) -- only R^2=0.42 on a CLEAN LINEAR target and 0.11 on a clean
        quadratic; badly undertrained at n~80, d=5280.
      * RBF kernel ridge with a median-distance gamma -- 0.02 on the clean linear
        target, and a POSITIVE permutation null.
      * PCA-24 + poly2 -- 0.36 on the clean linear target (over-parameterised).
    PCA first is what makes this work at d >> n: it reduces to a regime where a
    quadratic expansion is estimable, and the expansion is what buys the curvature.

    The synthetic used for tuning is LOW-RANK, matching the real states' measured
    participation ratio of 1.0-4.8 (D48). Isotropic synthetics were misleading in
    both directions and produced two wrong verdicts about the probe before that was
    noticed.

    The permutation null matters more here, not less: the quadratic expansion has
    more capacity, so a null at or above zero invalidates the figure.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    k = int(min(n_pc, max(2, len(x) // 4), x.shape[1]))
    mdl = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                        PolynomialFeatures(degree=2, include_bias=False),
                        StandardScaler(), Ridge(alpha=10.0))
    cv = (GroupKFold(n_splits=n_splits) if groups is not None
          else KFold(n_splits=n_splits, shuffle=True, random_state=0))

    def score(target):
        p = cross_val_predict(mdl, x, target, cv=cv, groups=groups)
        return float(1 - ((target - p) ** 2).sum() / ((target - target.mean()) ** 2).sum())

    r2 = score(y)
    rng = np.random.default_rng(seed)
    null = [score(rng.permutation(y)) for _ in range(n_null)]
    return r2, null
# ---8<---
