# Running remote GPU jobs: Kaggle and Yandex DataSphere

**Scope.** Operational procedure only — how to launch a remote run and get its data
back. This is not a fourth thinking-document (CLAUDE.md §6); findings go to
`claims_ledger.md`, open threads to `directions.md`, synthesis to `UNDERSTANDING.md`.
Every rule below is here because it cost this project a run.

---

## 0. The one-paragraph version

A remote run is observable when three things are true: it **verifies its own
instrument before producing numbers**, it **writes results to a path the platform
will actually collect**, and its bundle is **reproducible from the repo**. Two of
the three failed on 2026-08-09 within an hour of each other. Both were caught, but
only because the raw output was opened rather than the printed summary.

---

## 1. Self-check before science (CLAUDE.md §5, non-negotiable)

**A job that extracts an internal quantity must validate that extraction against
ground truth inside the same run, and halt if it fails.**

The QK-alignment probe reconstructs (q, k) from a `Wqkv` forward hook. Its
`verify_extraction` monkeypatches `scaled_dot_product_attention` for exactly one
forward, captures the tensors it is ACTUALLY called with, and asserts the
reconstruction matches to float32 precision. On first submission it did not:

```
self-check: max|q_hat - q_true| = 1.496e+01, max|k_hat - k_true| = 1.382e+01
SELF-CHECK FAILED -- stopping before any experimental number is produced.
```

Root cause: the spy captured the FIRST `scaled_dot_product_attention` call in the
forward and assumed it was `core_block[-1]`'s. A Huginn forward runs prelude (2
attention blocks) → core (r unrolls × 4 blocks) → coda (2), so the first call
belongs to the first PRELUDE block — an unrelated layer's weights. Fix: bracket the
capture with `register_forward_pre_hook`/`register_forward_hook` on the target
module so the spy arms only while that module's own forward is on the stack.

**Rules this implies.**

- The self-check runs FIRST and `return`s on failure. Never let it warn-and-proceed.
- Write the failure branch into the results file too (`{"passed": false}`), so a
  downstream reader cannot mistake a halted run for a missing one.
- **Reproduce the bug locally before resubmitting.** A tiny structural fake — the
  real module classes at tiny dimensions, no weight download — proved both that the
  old logic errored (q_err=1.59) and that the fix was exact (0.0), for zero GPU-cost.
  Do this rather than paying for a second remote run to find out.
- An error of ~15 is not a precision problem. Order-of-magnitude mismatches mean
  "wrong object", not "wrong dtype". Do not tune a tolerance to make it pass.

**Parallel rule for analysis scripts: a detection floor must be able to fire.**
`run_h0_within.py`'s planted-effect floor used 400 permutations for the main test
but a default of 40 for the floor, against a Bonferroni α of 0.00625. The minimum
achievable p-value with k permutations is 1/(k+1); 1/41 ≈ 0.024 can never clear
0.00625, so the floor reported 0% detection at every effect size including 1.0 sd.
That reads as "hopelessly underpowered design" and is actually "the check is
arithmetically incapable of returning anything". **Before trusting a null, confirm
its power curve is non-degenerate at the top end** — a floor that detects nothing
at a large planted effect is broken, not informative.

---

## 2. Getting the data back

### DataSphere

`outputs:` must be declared, the file must exist when the job ends, and the path
must be one the job actually wrote to.

```yaml
cmd: python job.py ${RESULT}
outputs:
  - qk_probe.json: RESULT
```

- **The output path arrives as a command-line argument.** Capture it at process
  start, `os.path.abspath` it, and write there. If the job `os.chdir`s (e.g. into a
  `git clone`), a relative path silently lands somewhere the platform never
  collects. This is exactly how the QK smoke run lost its per-unroll data while
  reporting `SUCCESS` — stdout survived, the JSON did not.
- **Do not add an `inputs:` block naming the entrypoint** when using
  `env.python.type: manual`; the script is included automatically and the CLI
  fails with `ValueError: Paths must be unique`.
- `outputs:` only uploads when the job ENDS. For long runs, prefer the tarball
  pattern (`scratch/ds_bank/cfg.yaml.template`): a `run.sh` that tars the whole
  working directory and `exit 0`s even on failure, so the log is recovered from a
  crashed run. Short jobs can write JSON directly.
- Retrieve with `--with-logs`; logs and declared outputs come separately:

```bash
datasphere project job download-files --id <JOB_ID> --with-logs --output-dir <DIR>
```

**Working GPU recipe on this account** (established by `datasphere_gpu_smoke` →
`datasphere_blayney`, do not re-derive):

```yaml
env:
  python:
    type: manual
    version: "3.11"
cloud-instance-type: gt4.1     # T4
```
with `pip install torch==2.5.1` pinned inside the job (needed for
`flex_attention`, capped by the driver at CUDA 12.2), then
`sed -i 's/<3.12/<3.13/' pyproject.toml` before `pip install -e .[model]`.
Verify `torch.cuda.is_available()` and print the device name before real work.

**CLI gotchas on this Mac.**

- `GRPC_DNS_RESOLVER=native` is REQUIRED on every `datasphere` invocation. The
  default c-ares resolver fails against the machine's VPN/Tailscale DNS server
  (100.98.255.254) although the OS resolver and `curl` work fine.
- `job get` takes `--id <ID>` alone — no `-p`. `job execute` takes both
  `-p <PROJECT_ID>` and `-c <CONFIG>`.
- Project: `bt12q57tmrs03pnt8drc`. Billing is the user's **personal** account
  (~80k RUB, valid to end of 2026), not the expired smiles2026 grant.

### Kaggle

- Kernel slugs are under `arsen4ikvar/`, not `arsenii324/` (the GitHub handle).
  A wrong slug returns a *permissions* error, which reads like a private-notebook
  problem and is not.
- **Two concurrent kernels maximum.** Build the bundle, commit it, and push the
  moment a slot frees rather than waiting.
- `kaggle kernels output <slug> -p <dir>` on a run with hundreds of raw arrays
  takes many minutes and exceeds a foreground timeout. Run it backgrounded.
- Bundles are committed (`.gitignore` negations) precisely so a "verified-live on
  GPU" claim has its producing code on disk; `docs/rigor_audit.md` records what
  happened when they were not.

---

## 3. Design the run wide, not the fleet parallel

**Two model instances will not fit on one T4.** Huginn-3.5B at float32 is ~14 GB
against 16 GB of device memory, so per-machine process parallelism OOMs. That is
not the axis to optimise.

The axis that pays: on a ~50-minute job, install + weight download + model load is
~10 minutes and is paid ONCE. Additional prompts, seeds, difficulty levels and
control arms inside the same forward loop are nearly free. The QK probe went from
24 forwards to 116 — adding 10× seeds *and* a whole control arm — for roughly 40
extra minutes and zero extra setup. **Prefer one wide job over two narrow ones**,
and fold the control into the same run as the treatment so both see identical
weights, dtype and driver.

Genuine parallelism belongs at the *platform* level: a DataSphere job and two
Kaggle kernels can run different experiments simultaneously, since they are
different machines.

This is also CLAUDE.md §2 ("instrument generously") in operational form: the extra
metrics and edge cases are where this project's real findings came from, and on a
remote job they are the cheapest thing on the bill.

---

## 4. Before you believe the result

- **Open the raw output, not the printed verdict** (CLAUDE.md §4). D93's verdict
  line said `NOT CONFIRMED`; the table two lines above it showed the position arm
  clearing α, and the floor beneath it showed 0% detection at 1.0 sd — the first
  worth flagging, the second a broken instrument. Neither was in the summary.
- Check the self-check line explicitly before reading any experimental number.
- Check the banked count (`banked N/M`) — a partial run reports SUCCESS.
- For a pre-registered analysis, run the script **unedited** first, record the
  result, and only then fix anything; note the fix and re-run separately so the
  pre-registered output and the corrected one are both in the record.
