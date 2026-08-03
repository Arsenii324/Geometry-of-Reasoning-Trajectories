"""Contraction rate rho ACROSS TRAINING: random init -> 8 checkpoints -> final model.

WHAT THIS DECIDES
    D42 claims training SLOWS the contraction (rho ~0.66 untrained -> ~0.91
    trained) and that this one parameter explains both effects in D41. Two
    weaknesses: (a) the untrained 0.662 is INFERRED from a readout curve via
    observable_convergence.py eq.(3), never measured on the operator; (b) two
    endpoints cannot distinguish "training raises rho" from "these two particular
    weight-sets happen to differ".

    tomg-group-umd published eight intermediate checkpoints of the SAME run --
    step-00006144 ... step-00041728, all with configs byte-identical to
    huginn-0125 (verified: n_embd=5280, mean_recurrence=32, same architecture).
    So rho can be measured as a FUNCTION OF TRAINING STEP. A monotone rise turns
    D42 from a two-point contrast into a trend; a flat or non-monotone curve
    falsifies it.

    All models measured by the SAME code, on the SAME prompts, in the SAME
    process, so differences are attributable to the weights alone.

    SECOND QUESTION, ANSWERED IN THE SAME RUN. D43 applies a rho measured on
    counting prompts to Geiping et al.'s GSM8K/ARC-C/HellaSwag saturation points.
    That transfer is D43's largest assumption (backlog 5.4). Prompts are therefore
    stratified across FOUR families -- counting, nesting depth, arithmetic word
    problem, commonsense continuation -- with token lengths deliberately spanning
    ~15 to ~74. If rho is a property of the OPERATOR it is near-constant across
    them; if it tracks the prompt, D43's bound must be restated per task and its
    cross-task comparison is invalid. The run prints that verdict explicitly.

TWO INDEPENDENT METHODS, because either alone has a known failure mode
    (A) TWO-ORBIT CONVERGENCE.  Same prompt, two different random initial
        latents; d_t = ||h_t^(1) - h_t^(2)|| contracts as rho^t. Measures the
        contraction of the MAP, independent of where the fixed point sits --
        the D24(7) method.
    (B) STEP-NORM DECAY.  s_t = ||h_{t+1} - h_t|| along one orbit, also ~ rho^t.
        Cheaper, but contaminated if the orbit has not reached the linear regime.
    Agreement is the internal check; disagreement is reported, not hidden.

THE FLOOR, WHICH BIASES ANY SUCH FIT IF IGNORED
    Finite arithmetic floors both curves; once there, decay stops, and including
    those points drags the slope toward zero and INFLATES rho. Points are kept
    only while above k * (empirical floor), floor = median of the last quarter --
    the rule in src/traj_geom/metrics/regime.py. n_used is reported, because a
    fit on three points is not a measurement. Validated locally against known
    rates 0.60-0.95: recovery bias +0.002..+0.015, always upward, so an
    UNDERstated difference, never an overstated one.

float32 throughout -- bfloat16 (unit roundoff 2^-8) would raise the floor far
enough to swallow the decay. Results are written after EVERY checkpoint, so a
timeout still yields a usable partial curve; the checkpoint order is chosen so
the most informative points land first.
"""

import gc
import json
import os
import random
import shutil
import subprocess


def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)


FINAL = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"

# (label, repo_id, training_step, revision). Revisions pinned 2026-08-04, the
# same provenance rule the rest of the project uses.
CKPT = "tomg-group-umd/step-{:08d}-recurrence_full_512_0"
SHA = {
    6144: "b5b1f9d44fdf3f1b91cb301dfaded77196e7bf3d",
    10752: "5e35596c8e8c79ecae57a8fab34625fe8ac640a8",
    11904: "aadca23e6829ea0bcf96b7965ebc61ad9fada17c",
    17920: "9c8576dc13cffdf93ec7da08eecf5b200bc8b47d",
    23808: "b531a3366a7d2a4171403038b785639606cb1297",
    29824: "252da2591e39f1ca27c063e018e254e470833c39",
    35840: "41580a97ec90f282e6e9b72f83073779808a53af",
    41728: "0fb03f39328917a92ca3059cc77b3e2115a35c94",
}
# Ordered so a timeout still leaves a usable curve: the two anchors and the
# extreme intermediates first.
SWEEP = [("untrained", None, 0, None), ("final", FINAL, 100000, REVISION)] + [
    (f"s{st:05d}", CKPT.format(st), st, SHA[st])
    for st in (6144, 41728, 17920, 29824, 10752, 23808, 35840, 11904)
]

M = 64
MAX_R = 128
N_PER_FAMILY = 3     # 4 families x 3 = 12 prompts per model, for the rho fits
N_EVAL = 60          # counting prompts for readout R^2 AND accuracy, per model
EVAL_R = 32          # Huginn's mean_recurrence; where the model is meant to be used
GEN_TOK = 3          # greedy tokens to decode when scoring accuracy
K_FLOOR = 3.0
TAIL_FRAC = 0.25


def prompts_by_family():
    """Four task families, so rho can be tested for CROSS-TASK CONSTANCY.

    D43 applies a rho measured on counting prompts to Geiping et al.'s
    GSM8K/ARC-C/HellaSwag saturation points. That transfer is the single largest
    assumption behind D43 (backlog 5.4). If rho is a property of the OPERATOR it
    should be near-constant across families; if it varies strongly with the
    prompt, D43's bound must be restated per-task and the cross-task comparison
    is invalid. Either way the answer is worth more than more counting prompts.
    """
    out = []
    for i in range(N_PER_FAMILY):
        rng = random.Random(i * 7919)
        bits = [1 if rng.random() < 0.5 else 0 for _ in range(M)]
        out.append(("counting",
                    "Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:"))
    for i in range(N_PER_FAMILY):
        rng = random.Random(i * 104729)
        seq, d = [], 0
        for _ in range(M // 2):
            if d == 0 or (rng.random() < 0.5 and d < 8):
                seq.append("(")
                d += 1
            else:
                seq.append(")")
                d -= 1
        seq += [")"] * d
        out.append(("nesting",
                    "String: " + " ".join(seq) + ". What is the maximum nesting depth? A:"))
    for i in range(N_PER_FAMILY):
        rng = random.Random(i * 15485863)
        a, b, c = rng.randint(11, 99), rng.randint(3, 19), rng.randint(2, 9)
        out.append(("arith",
                    f"A shop had {a} boxes. It sold {b} boxes each day for {c} days. "
                    f"How many boxes are left? A:"))
    stems = ["The man picked up the heavy suitcase and walked toward the platform. He",
             "She opened the oven, checked the bread, and decided it needed more time. Then she",
             "The dog heard the doorbell, ran into the hallway, and started barking. Next it"]
    for i in range(N_PER_FAMILY):
        out.append(("commonsense", stems[i % len(stems)]))
    return out


def fit_rho(curve, k=K_FLOOR, tail_frac=TAIL_FRAC):
    """Fit log(curve) linear in t over the pre-floor regime -> (rho, n_used, fit_r2)."""
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


def eval_set():
    """60 counting prompts with known totals, disjoint seeds from the rho prompts."""
    out = []
    for i in range(N_EVAL):
        rng = random.Random(500000 + i * 7919)
        p_one = (0.2, 0.35, 0.5, 0.65, 0.8)[i % 5]
        bits = [1 if rng.random() < p_one else 0 for _ in range(M)]
        out.append(("Sequence: " + " ".join(map(str, bits)) + ". How many ones? A:",
                    sum(bits)))
    return out


def readout_and_accuracy(model, tok, prompts):
    """Decodability AND capability on the SAME task, at the SAME depth.

    D41's first version paired a decodability measured here against an accuracy
    taken from a different configuration, and overclaimed as a result. Measuring
    both on the same prompts at every checkpoint is what makes "do decodability
    and capability move in opposite directions?" answerable instead of asserted.

    Accuracy is greedy generation + integer parse, matching the method behind
    results/counting_accuracy.csv, so the numbers are comparable to it.
    """
    import numpy as np
    import torch
    states, totals, correct = [], [], []
    for text, total in prompts:
        ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        cap = []
        mod = model.transformer.core_block[-1]
        mod._forward_hooks.clear()
        h = mod.register_forward_hook(
            lambda m, i, o, c=cap: c.append(o.detach()[0, -1, :].float().cpu().numpy()))
        try:
            torch.manual_seed(0)
            with torch.no_grad():
                out = model(input_ids=ids, num_steps=EVAL_R)
        finally:
            h.remove()
        states.append(cap[-1])
        totals.append(total)

        cur = ids
        gen = []
        logits = out.logits if hasattr(out, "logits") else out[0]
        for _ in range(GEN_TOK):
            nxt = int(logits[0, -1].argmax())
            gen.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
            with torch.no_grad():
                o2 = model(input_ids=cur, num_steps=EVAL_R)
            logits = o2.logits if hasattr(o2, "logits") else o2[0]
        txt = tok.decode(gen)
        digits = "".join(c for c in txt if c.isdigit() or c == " ").split()
        correct.append(bool(digits) and digits[0].isdigit() and int(digits[0]) == total)
        del ids, cur
        torch.cuda.empty_cache()

    x_states = np.stack(states).astype(np.float64)
    y = np.array(totals, float)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=1e3)), x_states, y,
                             cv=KFold(5, shuffle=True, random_state=0))
    r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return {"readout_r2": r2, "mean_abs_err": float(np.abs(y - pred).mean()),
            "accuracy": float(np.mean(correct)), "n": len(y)}


def orbit(model, ids, seed, max_r):
    """h_1..h_max_r at the final position, from a controlled random initial latent."""
    import numpy as np
    import torch
    cap = []
    mod = model.transformer.core_block[-1]
    mod._forward_hooks.clear()
    h = mod.register_forward_hook(
        lambda m, i, o: cap.append(o.detach()[0, -1, :].float().cpu().numpy()))
    try:
        torch.manual_seed(seed)      # controls h_0
        with torch.no_grad():
            model(input_ids=ids, num_steps=max_r)
    finally:
        h.remove()
    return np.stack(cap)


def measure(model, tok, label, prompts):
    import numpy as np
    import torch
    rows = []
    for i, (family, p) in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        a = orbit(model, ids, 1000 + i, MAX_R)
        b = orbit(model, ids, 2000 + i, MAX_R)
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
              f"rho_step={rs:.4f}(n={ns},fit {fs:.3f})  ||h||={rows[-1]['norm_h']:.2f}",
              flush=True)
        del ids
        torch.cuda.empty_cache()
    return rows


def summarise(rows):
    import numpy as np
    def agg(rs, key):
        v = np.array([r[key] for r in rs], float)
        v = v[np.isfinite(v)]
        return {"mean": float(v.mean()) if len(v) else float("nan"),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                "n": int(len(v))}
    out = {k: agg(rows, k) for k in ("rho_orbit", "rho_step")}
    out["by_family"] = {}
    for fam in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] == fam]
        out["by_family"][fam] = {k: agg(rs, k) for k in ("rho_orbit", "rho_step")}
    return out


def purge(repo_id):
    """Drop the HF cache for one repo -- 10 x ~7GB would otherwise exhaust the disk."""
    if not repo_id:
        return
    root = os.path.expanduser("~/.cache/huggingface/hub")
    d = os.path.join(root, "models--" + repo_id.replace("/", "--"))
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print(f"    purged cache {d}", flush=True)


def main():
    run("pip install -q 'transformers>=4.50,<4.54' scipy")
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    print("CUDA:", torch.cuda.is_available(), flush=True)

    tok = AutoTokenizer.from_pretrained(FINAL, revision=REVISION)
    prompts = prompts_by_family()
    print(f"{len(prompts)} prompts across "
          f"{len(set(f for f, _ in prompts))} families; token lengths "
          f"{sorted({len(tok(p).input_ids) for _, p in prompts})}", flush=True)

    eval_prompts = eval_set()
    print(f"eval set: {len(eval_prompts)} counting prompts, totals "
          f"{min(t for _, t in eval_prompts)}..{max(t for _, t in eval_prompts)}",
          flush=True)

    cfg = AutoConfig.from_pretrained(FINAL, revision=REVISION, trust_remote_code=True)
    results = {}

    for label, repo, step, rev in SWEEP:
        print(f"\n=== {label} (step {step}) ===", flush=True)
        model = None
        try:
            if repo is None:
                torch.manual_seed(0)
                model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    repo, revision=rev, trust_remote_code=True)
            model = model.to(torch.float32).to("cuda").eval()
            rows = measure(model, tok, label, prompts)
            results[label] = {"step": step, "repo": repo, "revision": rev, "rows": rows,
                              "summary": summarise(rows)}
            with open("rho_vs_training.json", "w") as f:   # rho is the primary result;
                json.dump(results, f, indent=1)            # bank it before the eval pass
            try:
                ev = readout_and_accuracy(model, tok, eval_prompts)
                results[label]["eval"] = ev
                print(f"  {label} eval: readout R2={ev['readout_r2']:+.4f}  "
                      f"err={ev['mean_abs_err']:.3f}  accuracy={ev['accuracy']:.1%}",
                      flush=True)
            except Exception as e:                                # noqa: BLE001
                print(f"  {label} eval FAILED: {type(e).__name__}: {str(e)[:150]}",
                      flush=True)
                results[label]["eval_error"] = f"{type(e).__name__}: {e}"
            s = results[label]["summary"]
            print(f"  {label}: rho_orbit={s['rho_orbit']['mean']:.4f}"
                  f"+-{s['rho_orbit']['sd']:.4f}  "
                  f"rho_step={s['rho_step']['mean']:.4f}+-{s['rho_step']['sd']:.4f}",
                  flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {label} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            results[label] = {"step": step, "repo": repo, "revision": rev,
                              "error": f"{type(e).__name__}: {e}"}
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            purge(repo)
            with open("rho_vs_training.json", "w") as f:      # partial-safe
                json.dump(results, f, indent=1)

    print("\n=== rho VERSUS TRAINING STEP ===")
    ok = [(v["step"], k, v["summary"]) for k, v in results.items() if "summary" in v]
    ok.sort()
    print(f"  {'step':>8} {'label':>10} {'rho_orbit':>18} {'rho_step':>18}")
    for st, lab, s in ok:
        print(f"  {st:>8} {lab:>10} {s['rho_orbit']['mean']:>10.4f}"
              f" +-{s['rho_orbit']['sd']:<6.4f} {s['rho_step']['mean']:>10.4f}"
              f" +-{s['rho_step']['sd']:<6.4f}")
    if len(ok) >= 4:
        from scipy.stats import spearmanr
        steps = np.array([o[0] for o in ok], float)
        for key in ("rho_orbit", "rho_step"):
            vals = np.array([o[2][key]["mean"] for o in ok], float)
            m = np.isfinite(vals)
            if m.sum() >= 4:
                rho_s, p = spearmanr(steps[m], vals[m])
                print(f"\n  spearman(training step, {key}) = {rho_s:+.4f}, p = {p:.4g}"
                      f"   [n={int(m.sum())} checkpoints]")
        print("\n  D42 SURVIVES if rho rises with training step. A flat or")
        print("  non-monotone curve falsifies 'training slows the contraction'.")

    print("\n=== DOES CAPABILITY RISE WHILE DECODABILITY FALLS? (D41(3c)) ===")
    ev = [(v["step"], k, v["eval"]) for k, v in results.items() if "eval" in v]
    ev.sort()
    if ev:
        print(f"  {'step':>8} {'label':>10} {'readout R2':>11} {'abs err':>9} {'accuracy':>9}")
        for st, lab, e in ev:
            print(f"  {st:>8} {lab:>10} {e['readout_r2']:>+11.4f} "
                  f"{e['mean_abs_err']:>9.3f} {e['accuracy']:>8.1%}")
        if len(ev) >= 4:
            from scipy.stats import spearmanr
            st = np.array([e[0] for e in ev], float)
            acc = np.array([e[2]["accuracy"] for e in ev], float)
            r2s = np.array([e[2]["readout_r2"] for e in ev], float)
            ra, pa = spearmanr(st, acc)
            rr, pr = spearmanr(st, r2s)
            print(f"\n  spearman(step, accuracy)   = {ra:+.4f}, p={pa:.4g}")
            print(f"  spearman(step, readout R2) = {rr:+.4f}, p={pr:.4g}")
            if ra > 0 and rr < 0:
                print("  => accuracy RISES while decodability FALLS. 'Opposite directions'")
                print("     is earned on matched data and D41(3) can be restated.")
            elif acc.max() == 0:
                print("  => NO model scores above zero at M=64, so this task cannot")
                print("     support a capability contrast at all. D41(3)'s retraction")
                print("     stands and the claim needs an easier task.")
            else:
                print("  => the two do not move oppositely; D41(3) stays retracted.")

    print("\n=== IS rho CONSTANT ACROSS TASK FAMILIES? (backlog 5.4, D43's key assumption) ===")
    fams = sorted({f for _, v in results.items() if "rows" in v for f in
                   {r["family"] for r in v["rows"]}})
    print(f"  {'label':>10} " + " ".join(f"{f:>13}" for f in fams) + "   spread")
    spreads = []
    for _st, lab, sm in ok:
        bf = sm.get("by_family", {})
        vals = [bf.get(f, {}).get("rho_orbit", {}).get("mean", float("nan")) for f in fams]
        fin = [v for v in vals if np.isfinite(v)]
        sp = (max(fin) - min(fin)) if len(fin) > 1 else float("nan")
        spreads.append(sp)
        print(f"  {lab:>10} " + " ".join(f"{v:>13.4f}" for v in vals) + f"   {sp:>7.4f}")
    fin_sp = [x for x in spreads if np.isfinite(x)]
    if fin_sp:
        print(f"\n  max across-family spread within a model: {max(fin_sp):.4f}")
        print("  the effect D42 claims (untrained -> trained): ~0.25")
        if max(fin_sp) < 0.05:
            print("  => rho is a property of the OPERATOR, not the prompt. D43's")
            print("     cross-task transfer is licensed and backlog 5.4 closes.")
        else:
            print("  => rho varies materially with the prompt. D43's bound must be")
            print("     restated PER TASK and its cross-task comparison is invalid.")


main()
