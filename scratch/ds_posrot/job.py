"""A46: is the rotation regime a property of the PROMPT, or of the one position we measure?

THE GAP, FOUND IN THEIR FIGURES AND NOT IN OUR DATA. Every regime measurement in this project
reads **one token position**: `core_block[-1]` output at `[0, -1]`, the last prompt position.
D132, D134, D141, D161, D164, D167, D173, D176, D185 all rest on that single sample. The Huginn
paper's own orbit figures are drawn at **interior** positions -- their rotating example is the
token `3` inside *"Claire makes a 3 egg omelette"*, not the final token (D187d).

So the regime may be a property of the prompt, as we have written it, or a property of the
position we happen to sample. **If interior positions rotate freely inside prompts whose last
position settles, then D141's threshold was fitted on an unrepresentative population and every
"this prompt rotates" statement needs re-scoping.** That is a large claim about our own record
and it is one forward pass per prompt to test.

WHY IT IS CHEAP. The hook already sees the full `[T, 5280]` state at every unroll; we have
simply been indexing `[-1]`. Keeping all positions and computing `rotation_power` per position
costs no extra forwards -- only the arithmetic.

PREREGISTERED PREDICTIONS:

  P1  REPLICATION GATE, VOID WITHOUT IT. The LAST position's R must reproduce the banked value
      for that prompt: `symbol` ~0.77-0.85 (rotating), `element` ~0.05-0.07 (settling), against
      D141's threshold 0.6677. If the last position does not reproduce, this run is not
      measuring the same object as the rows above.

  P2  PRIMARY -- HOW MANY POSITIONS ROTATE? For a prompt our record calls ROTATING, what
      fraction of its token positions exceed the threshold? For one we call SETTLING, the same.
      Registered before running: **if "rotating" prompts rotate at nearly all positions and
      "settling" prompts at nearly none, the regime is a property of the prompt and our record
      stands as written.** If the two overlap substantially, it is a property of the position
      and every regime row needs a scope line.

  P3  WHERE, STRUCTURALLY. R as a function of position index, printed per prompt. The
      instruction noun sits at a known offset, so if rotation is localised around it that is
      visible directly rather than inferred.

  P4  THEIR CASE, ON OUR INSTRUMENT. A word-problem prompt in the shape of their example
      (*"Claire makes a 3 egg omelette every morning..."*) is included. If interior digit
      tokens rotate while its last position settles, we reproduce their observation and confirm
      the gap is real; if nothing rotates anywhere, our instrument and theirs disagree and that
      must be resolved before either is quoted against the other.

  P5  A SETTLING PROMPT IS THE CONTROL FOR P4. `element` in the same template. Any position-wise
      rotation there is either a real second population or a threshold artefact, and the
      distribution of R over all positions is printed so the reader can see which.
"""

import json
import os
import subprocess
import sys
import time

NUM_STEPS = 64
SEED = 20260811
THRESHOLD = 0.6677          # D141
MARKER = "A"
SEQS = ("1 6 0 1 7 7 8 1", "3 5 5 3 6 1 5 2", "2 4 4 1 7 5 0 9")
ROT_NOUNS = ("symbol", "symptom", "cymbal")      # D134's rotating set
SET_NOUNS = ("element", "token", "digit")        # D134's settling set
OUTDIR = os.path.abspath("out")
WALL_BUDGET_S = 9000

# P4: their own example shape, and two more free-form prompts of the kind we never test
THEIRS = (
    "Claire makes a 3 egg omelette every morning for breakfast. How many eggs will "
    "she eat in 4 weeks?",
    "What do you think of Goethe's Faust?",
    "What is the capital of France?",
)


def build_items(rng=None):
    out = []
    for n in ROT_NOUNS:
        for i, s in enumerate(SEQS):
            out.append({"kind": "rot_noun", "label": n, "item": i, "prompt": prompt_for(n, s)})
    for n in SET_NOUNS:
        for i, s in enumerate(SEQS):
            out.append({"kind": "set_noun", "label": n, "item": i, "prompt": prompt_for(n, s)})
    for i, p in enumerate(THEIRS):
        out.append({"kind": "freeform", "label": f"theirs{i}", "item": i, "prompt": p})
    return out


def prompt_for(word, seq, mk=MARKER):
    rules = (f"for task A, report the largest {word} of the sequence. "
             f"For task B, report the smallest {word} of the sequence.")
    return f"Rules: {rules}\nSequence: {seq}\nTask: {mk}"


def run(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main():
    out_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
        os.path.abspath("posrot.json")
    mnt = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"results -> {out_path}; weights mount -> {mnt}", flush=True)

    run("git clone -b claude/geometry-reasoning-recap-rhe0bp "
        "https://github.com/Arsenii324/Geometry-of-Reasoning-Trajectories.git repo")
    os.chdir("repo")
    run("pip install torch==2.5.1")
    run("sed -i 's/<3.12/<3.13/' pyproject.toml")
    run("pip install -e .[model]")
    sys.path.insert(0, os.path.abspath("src"))

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from traj_geom.metrics.dynamics import rotation_power

    print("CUDA:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    tok = AutoTokenizer.from_pretrained(mnt)
    cfg = AutoConfig.from_pretrained(mnt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mnt, config=cfg, trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()

    core_last = model.transformer.core_block[-1]

    def all_position_R(prompt):
        """R at EVERY token position. Same hook as always; we stop indexing [-1]."""
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(model.device)
        traj = []
        core_last._forward_hooks.clear()

        def hook(_m, _i, o):
            traj.append(o.detach()[0].float().cpu().numpy().copy())   # [T, 5280]

        h = core_last.register_forward_hook(hook)
        try:
            with torch.no_grad():
                torch.manual_seed(SEED)
                model(input_ids=ids, num_steps=NUM_STEPS)
        finally:
            h.remove()
        T = np.stack(traj, axis=0)                     # [unrolls, T, 5280]
        toks = [tok.decode([int(i)]) for i in ids[0]]
        return [float(rotation_power(T[:, p, :])) for p in range(T.shape[1])], toks

    items = build_items()
    print(f"{len(items)} prompts", flush=True)
    t0, rows = time.time(), []
    for n, it in enumerate(items):
        try:
            Rs, toks = all_position_R(it["prompt"])
            ok, why = True, ""
        except Exception as exc:  # noqa: BLE001
            Rs, toks, ok, why = [], [], False, f"{type(exc).__name__}: {exc}"
        rows.append({**it, "ok": ok, "why": why, "R_by_pos": Rs, "tokens": toks,
                     "n_tokens": len(Rs),
                     "R_last": Rs[-1] if Rs else None,
                     "n_rot": sum(1 for r in Rs if r > THRESHOLD),
                     "frac_rot": (sum(1 for r in Rs if r > THRESHOLD) / len(Rs)) if Rs else None})
        if n % 4 == 0:
            print(f"  {n}/{len(items)} ({time.time() - t0:.0f}s) {it['label']}: "
                  f"last {rows[-1]['R_last']}, {rows[-1]['n_rot']}/{rows[-1]['n_tokens']} pos rotate",
                  flush=True)
            with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
                json.dump(rows, f)
        if time.time() - t0 > WALL_BUDGET_S:
            print("WALL BUDGET -- banking and stopping cleanly", flush=True)
            break

    ok_rows = [r for r in rows if r["ok"]]
    import statistics as st
    print("\n=== P1 GATE: last position must reproduce the record ===", flush=True)
    for kind in ("rot_noun", "set_noun"):
        v = [r["R_last"] for r in ok_rows if r["kind"] == kind]
        if v:
            print(f"  {kind:9s} last-position R: median {st.median(v):.3f} "
                  f"range {min(v):.3f}-{max(v):.3f}  "
                  f"{sum(1 for x in v if x > THRESHOLD)}/{len(v)} above threshold", flush=True)

    print("\n=== P2 PRIMARY: what fraction of POSITIONS rotate? ===", flush=True)
    print(f"{'kind':10s} {'label':10s} {'n_tok':>6s} {'R_last':>7s} {'pos rotating':>13s}")
    for r in ok_rows:
        print(f"{r['kind']:10s} {r['label']:10s} {r['n_tokens']:6d} {r['R_last']:7.3f} "
              f"{r['n_rot']:5d}/{r['n_tokens']:<5d} = {r['frac_rot']:.2f}", flush=True)
    for kind in ("rot_noun", "set_noun", "freeform"):
        v = [r["frac_rot"] for r in ok_rows if r["kind"] == kind]
        if v:
            print(f"  {kind:9s}: mean fraction of positions rotating = {st.mean(v):.3f} "
                  f"(range {min(v):.3f}-{max(v):.3f})", flush=True)

    print("\n=== P3/P4 STRUCTURE: R by position, with tokens ===", flush=True)
    for r in ok_rows:
        if r["kind"] == "freeform" or (r["item"] == 0 and r["label"] in ("symbol", "element")):
            print(f"  [{r['label']}] last R {r['R_last']:.3f}, {r['n_rot']}/{r['n_tokens']} rotate",
                  flush=True)
            print("     " + "  ".join(
                f"{t!r}:{x:.2f}" + ("*" if x > THRESHOLD else "")
                for t, x in list(zip(r["tokens"], r["R_by_pos"]))[:26]), flush=True)

    with open(out_path, "w") as f:
        json.dump({"rows": rows, "threshold": THRESHOLD, "seed": SEED,
                   "num_steps": NUM_STEPS, "elapsed_s": time.time() - t0}, f)
    print(f"DONE {len(rows)} prompts, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
