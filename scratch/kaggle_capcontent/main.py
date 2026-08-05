"""Is "content is architectural" an artefact of only testing tasks the model fails?

THE BLIND SPOT (UNDERSTANDING.md 6.1)
    Every result behind this project's headline -- that training changes the
    model's DYNAMICS and not the CONTENT of its state -- was measured on
    counting-family tasks where the trained model scores ~0% (D56, D60, D61):

        D40  untrained reproduces every endpoint metric
        D41  untrained decodes the total 19x more precisely
        D48  untrained count representation is literally 1-dimensional (PR 1.0 vs 4.8)
        D53  untrained carries the register slightly better (0.7498 vs 0.7175)

    If the model cannot perform the task then there is NOTHING FOR TRAINING TO
    HAVE ENCODED, and discovering that random weights represent the input equally
    well is close to tautological -- both arms are carrying an unused input. The
    headline may be an artefact of never testing content where content exists.

THE DESIGN: CAPABILITY AS A MODERATOR
    Rather than argue about it, span capability and see whether the content gap
    tracks it. Four tasks chosen to run from certain success to certain failure:

        echo_num    repeat a 2-digit number   expected ~100%: copy is 100% (D60)
        add1        single-digit addition     expected high; a 3.5B LM does this
        count4      count ones in 4 bits      25% at n=4 (D56)
        count16     count ones in 16 bits     0% from n=8 onward (D56)

    EVERY PROBE TARGET IS AN INTEGER ON A COMPARABLE SCALE. An earlier draft used
    string `copy` with the word encoded base-26 as the latent; that encoding is
    arbitrary and essentially not linearly decodable, so its R2 would not have been
    comparable to the integer tasks and the moderation test would have been
    meaningless. `echo_num` keeps the high-capability cell while making its latent a
    plain integer. String `copy` is retained as the harness control only, scored for
    accuracy and never probed.

    For each task and each arm we measure TWO things on the SAME forward pass:
      CAPABILITY  greedy accuracy, chat-formatted
      CONTENT     cross-validated decodability of the task's latent variable from
                  the answer-position state (ridge, permutation null, grouped folds)

    PRE-REGISTERED PREDICTION, written before running:
      If 6.1 is a real confound, `content_gap = R2(trained) - R2(untrained)` is
      POSITIVE where capability is high and ~0 or negative where capability is 0 --
      i.e. corr(capability, content_gap) > 0. That would scope the headline to
      "training does not change content ON TASKS THE MODEL CANNOT DO", which is a
      much weaker and more honest claim.
      If instead the content gap is ~0 or negative EVERYWHERE, including on `copy`
      where the trained model is at 100% and the untrained at 0%, then the headline
      SURVIVES its most serious challenge and is considerably strengthened.

    Either outcome is decisive, which is why this is worth a kernel.

CONTROLS
    * `copy` doubles as the harness control: if trained accuracy on it is not high,
      the generator is broken and nothing may be read (the D62 lesson).
    * The permutation null must be NEGATIVE for every decodability figure, since
      d=5280 >> n (D41(6)).
    * Both arms see identical prompts in the same process.
"""
# ruff: noqa: E402  -- inlined blocks necessarily precede the body's imports
# ---- BUILT by scripts/build_kernel.py from scratch/_lib/kernel_common.py.
# ---- Edit body.py and rebuild; edits to this file are overwritten.
# ---- inlined blocks: run load_arm free_arm batched_generate assert_generation_works cv_r2 cv_r2_nonlinear


def run(cmd):
    """Shell out, echoing the command so the Kaggle log shows what was installed."""
    import subprocess
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


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

import json
import random

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
NUM_STEPS = 32
N_ITEMS = 80          # enough for a ridge probe with a permutation null
MAX_NEW = 8

TASKS = ("echo_num", "add1", "count4", "count16")
CONTROL = "copy"


def item(task, seed):
    """(question_body, gold_string, latent_value_for_the_probe)."""
    rng = random.Random(seed * 7919 + hash(task) % 997)
    if task == "copy":
        w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        return (f"Repeat this word exactly.\nWord: {w}", w, 0.0)   # control, not probed
    if task == "echo_num":
        v = rng.randint(10, 99)
        return (f"Repeat this number exactly.\nNumber: {v}", str(v), float(v))
    if task == "add1":
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        return (f"What is {a} + {b}?", str(a + b), float(a + b))
    n = 4 if task == "count4" else 16
    bits = [rng.randint(0, 1) for _ in range(n)]
    return (f"Count how many ones are in this sequence.\nSequence: "
            f"{' '.join(map(str, bits))}", str(sum(bits)), float(sum(bits)))


def render(body, tok):
    return tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True)


def score(task, pred, gold):
    p = pred.lower().strip()
    if task == "copy":
        return float(gold in "".join(c for c in p if c.isalpha()))
    nums, cur = [], ""
    for c in p:
        if c.isdigit():
            cur += c
        else:
            if cur:
                nums.append(cur)
            cur = ""
    if cur:
        nums.append(cur)
    return float(bool(nums) and nums[-1] == gold)


def answer_states(model, tok, texts):
    """Final-unroll state at the answer position, one row per prompt."""
    import numpy as np
    import torch
    rows = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        cap = []
        mod = model.transformer.core_block[-1]
        mod._forward_hooks.clear()
        h = mod.register_forward_hook(
            lambda m, i, o, c=cap: c.append(o.detach()[0, -1, :].float().cpu().numpy()))
        try:
            torch.manual_seed(0)
            with torch.no_grad():
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        rows.append(cap[-1])
        del ids
        torch.cuda.empty_cache()
    return np.stack(rows).astype(np.float64)


def main():
    run("pip install -q 'transformers>=4.50,<4.54' scikit-learn scipy")
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    data = {t: [item(t, s) for s in range(N_ITEMS)] for t in (*TASKS, CONTROL)}
    for t in (*TASKS, CONTROL):
        b, g, y = data[t][0]
        print(f"  {t:>8}: {b.replace(chr(10), ' | ')!r} -> {g!r} (latent {y})", flush=True)

    res = {}
    for arm in ("trained", "untrained"):
        print(f"\n=== {arm} ===", flush=True)
        model = None
        try:
            model = load_arm(None if arm == "untrained" else MODEL_ID, cfg,
                             0 if arm == "untrained" else REVISION)
            if arm == "trained":
                assert_generation_works(model, tok)
            res[arm] = {}
            for t in (*TASKS, CONTROL):
                texts = [render(b, tok) for b, _, _ in data[t]]
                preds = batched_generate(model, tok, texts, max_new=MAX_NEW,
                                         num_steps=NUM_STEPS, verbose=False)
                acc = float(np.mean([score(t, p, g)
                                     for p, (_, g, _) in zip(preds, data[t], strict=True)]))
                if t == CONTROL:                      # harness control: accuracy only
                    res[arm][t] = {"acc": acc, "r2": float("nan"),
                                   "null_max": float("nan"), "sample": preds[0][:40]}
                    print(f"  {t:>8}: acc {acc:>6.1%}   (control, not probed)", flush=True)
                    continue
                x = answer_states(model, tok, texts)
                y = np.array([v for _, _, v in data[t]], float)
                r2, null = cv_r2(x, y, n_null=12)
                r2n, nulln = cv_r2_nonlinear(x, y, n_null=6)
                res[arm][t] = {"acc": acc, "r2": r2, "null_max": float(max(null)),
                               "null_mean": float(np.mean(null)),
                               "r2_nonlin": r2n, "null_nonlin_max": float(max(nulln)),
                               "sample": preds[0][:40]}
                print(f"  {t:>8}: acc {acc:>6.1%}   linear R2 {r2:>+7.4f} (null "
                      f"{max(null):>+6.3f})   nonlinear R2 {r2n:>+7.4f} (null "
                      f"{max(nulln):>+6.3f})   e.g. {preds[0][:22]!r}", flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {arm} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        finally:
            free_arm(model, None if arm == "untrained" else MODEL_ID)
            with open("cap_content.json", "w") as f:
                json.dump(res, f, indent=1)

    print("\n=== DOES THE CONTENT GAP TRACK CAPABILITY? (UNDERSTANDING 6.1) ===")
    if "trained" not in res or "untrained" not in res:
        print("  an arm failed; nothing may be concluded")
        return
    t_, u_ = res["trained"], res["untrained"]
    if t_.get(CONTROL, {}).get("acc", 0) < 0.5:
        print(f"  HARNESS CONTROL FAILED: trained {CONTROL} = "
              f"{t_.get(CONTROL, {}).get('acc')}")
        print("  no conclusion may be drawn (the D62 lesson).")
        return
    print(f"  {'task':>8} {'cap(trained)':>13} {'cap(untr)':>10} {'R2 trained':>11} "
          f"{'R2 untr':>9} {'content gap':>12}")
    caps, gaps = [], []
    for t in TASKS:
        ct, cu = t_[t]["acc"], u_[t]["acc"]
        gap = t_[t]["r2"] - u_[t]["r2"]
        caps.append(ct)
        gaps.append(gap)
        print(f"  {t:>8} {ct:>12.1%} {cu:>9.1%} {t_[t]['r2']:>+11.4f} "
              f"{u_[t]['r2']:>+9.4f} {gap:>+12.4f}")
    print("\n  === and is the content NONLINEARLY encoded in the trained arm? ===")
    print("  (every content measure in this project is a LINEAR probe; if training")
    print("   encodes the same content nonlinearly, a linear probe favours the")
    print("   untrained arm for reasons unrelated to information content)")
    print(f"  {'task':>8} {'lin gap':>9} {'nonlin gap':>11} {'trained lin->nonlin':>20}")
    for t in TASKS:
        lg = t_[t]["r2"] - u_[t]["r2"]
        ng = t_[t].get("r2_nonlin", float("nan")) - u_[t].get("r2_nonlin", float("nan"))
        lift = t_[t].get("r2_nonlin", float("nan")) - t_[t]["r2"]
        print(f"  {t:>8} {lg:>+9.4f} {ng:>+11.4f} {lift:>+20.4f}")
    lifts = [t_[t].get("r2_nonlin", float("nan")) - t_[t]["r2"] for t in TASKS]
    lifts_u = [u_[t].get("r2_nonlin", float("nan")) - u_[t]["r2"] for t in TASKS]
    print(f"  mean nonlinear lift: trained {np.nanmean(lifts):+.4f}, "
          f"untrained {np.nanmean(lifts_u):+.4f}")
    if np.nanmean(lifts) > np.nanmean(lifts_u) + 0.05:
        print("  => the TRAINED arm gains more from a nonlinear probe. The linear-only")
        print("     content measures underestimate it, and D41/D48/D53 must be re-read.")
    else:
        print("  => no differential nonlinear lift. The linear probes were not the")
        print("     reason the untrained arm looked better.")

    from scipy.stats import spearmanr
    rho, p = spearmanr(caps, gaps)
    print(f"\n  spearman(capability, content gap) = {rho:+.3f}, p={p:.3f}  (n={len(caps)})")
    hi = [g for c, g in zip(caps, gaps, strict=True) if c > 0.5]
    print(f"  mean content gap where capability > 50%: "
          f"{np.mean(hi) if hi else float('nan'):+.4f}")
    if hi and np.mean(hi) > 0.05:
        print("  => 6.1 IS A REAL CONFOUND. Training DOES change content where the")
        print("     model can do the task. The headline must be scoped to tasks the")
        print("     model cannot perform.")
    else:
        print("  => 6.1 IS NOT A CONFOUND. The content gap stays ~0 even where the")
        print("     trained model is at 100% and the untrained at 0%. The headline")
        print("     survives its most serious challenge.")


main()
