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

**`pip install -e .` can succeed and still leave the package unimportable.**
Under `env.python.type: manual`, `pip` and `python` are not guaranteed to be the
same environment. A job died on

```
ModuleNotFoundError: No module named 'traj_geom'
```

*after* pip printed `Successfully installed ... traj-geom-0.1.0` in the same log.
Fix — put the source root on the path directly, independent of which interpreter
pip chose:

```python
sys.path.insert(0, os.path.abspath("src"))   # after os.chdir into the clone
```

Only jobs that import the project package hit this; one that uses nothing but
torch/transformers will pass and give false confidence that the install works.

**`Killed` in the log means the OOM killer, and the traceback above it may be a red
herring.** A control run died with a HuggingFace `RemoteDisconnected` visible in stderr
— but the very next lines showed `Retrying in 1s [Retry 1/5]` and then a successful
download. The disconnect was **not** the cause. The cause was three lines later:

```
bash: line 1:   109 Killed                  ( python job.py /job/result_gwx )
```

`Killed` is SIGKILL from the Linux OOM killer, i.e. **host RAM**. A *GPU* OOM raises a
Python `torch.cuda.OutOfMemoryError` with a traceback and never appears as `Killed`.
Diagnosing this as a network failure cost one wasted relaunch.

The specific trap: `AutoModelForCausalLM.from_pretrained(..., low_cpu_mem_usage=True)`
streams shards directly to the GPU, but **`from_config` has no such path** — it
materialises the whole model in float32 **on CPU** first (~14 GB for 3.5B), which the
pod cannot take. Build directly on the device instead:

```python
with torch.device("cuda"):
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
```

The model's own `_init_weights` still runs under `post_init`, so a randomly-initialised
control keeps the architecture's real initialisation scheme rather than uninitialised
memory. Print `free -g` and `torch.cuda.memory_allocated()` around each load so the next
OOM is diagnosable from the log alone.

**Partial results survive an errored job — check before re-running.** DataSphere collected
the declared `outputs:` file even though the job ended in ERROR, because the script wrote
it incrementally after each arm. The completed arm's data was fully recoverable. There is
no resume: `job attach` only works on a *running* job and `job fork` starts a fresh run
from a template, so **write results incrementally rather than once at the end.**

**CLI gotchas on this Mac.**

- `GRPC_DNS_RESOLVER=native` is REQUIRED on every `datasphere` invocation. The
  default c-ares resolver fails against the machine's VPN/Tailscale DNS server
  (100.98.255.254) although the OS resolver and `curl` work fine.
- **DataSphere project ID: `bt12q57tmrs03pnt8drc`.** Recorded here because it was
  written down nowhere and recovering it cost ~15 minutes: `job get` does not
  report it, the CLI logs under `$TMPDIR/datasphere/` do not contain it, `yc` has
  no `datasphere` subcommand in this install, and `job list` REQUIRES it. It was
  finally found by `git grep` over history. A neighbouring id, `bt14qn4u9t3n09nfjoqu`,
  appears in a sibling directory's launch command and returns PERMISSION_DENIED —
  it is not this account's.
- **DataSphere's CLI IMPORTS your job.py locally** to analyse dependencies, so a
  module-level `import numpy` fails with `ModuleNotFoundError` in the pipx venv
  before anything uploads. **All third-party imports go inside `main()`, after the
  pip installs** -- which is why every working kernel here does that. This does NOT
  contradict D125's "imports at the top": that rule is about position INSIDE
  `main()`, putting REPO imports right after the editable install so a missing
  symbol fails in 30 s instead of after every forward has run.
- **DataSphere requires `if __name__ == '__main__':` in the main script.** A Kaggle
  kernel ending in a bare `main()` is rejected before upload with
  `ValueError: Main script must have line ...`. Costs seconds, but it is the first
  thing to hit when porting a Kaggle kernel across.
- **Launch from the config's own directory.** `cmd: python job.py` is resolved
  relative to the CWD, so `-c scratch/foo/config.yaml` from the repo root dies with
  `FileNotFoundError: job.py` before contacting the API.
- **Before a long Kaggle run, compute whether it CAN finish, from the first completed
  stage.** CLRS: depth 4 took 6126s for 222 items, so depth 32 (8x the unrolls) needs
  ~49005s and the run dies 3.4h short of finishing it. That was knowable the moment
  depth 4 printed, at 6470s of a 43200s budget — 10 hours before the loss. The check is
  one line of arithmetic against the first stage's wall time; do it then, not at hour 10.
- **And write output INCREMENTALLY, per batch, not per stage.** A `json.dump` placed after
  a whole depth completes saves nothing when the stage itself is what overruns.
- **What Kaggle does with output on an abnormal end is UNVERIFIED here.** Measured
  2026-08-09: `kaggle kernels output <slug>` on a RUNNING kernel returns **zero files**,
  so output is collected at the end of a run, not streamed. Whether a timeout kill or a
  manual stop still collects `/kaggle/working` was **not** established — an earlier
  revision of this file asserted it did not, which was inference stated as fact and is
  retracted. Design so it does not matter: if a run's value depends on output surviving
  an abnormal end, the run is already badly designed.
- **PUSH BEFORE YOU LAUNCH.** Remote jobs `git clone` the branch from GitHub, so a
  kernel importing anything added locally fails on the REMOTE's older copy. On
  2026-08-09 A4b died at `ImportError: cannot import name 'require_null_can_move'`
  with the branch **31 commits ahead of origin**. It cost nothing only because the
  import sat in the analysis block, after all 256 forwards had run, and incremental
  saves preserved 225 records. At the top of the file it would have lost the entire
  run before the first forward. `git status -sb` must show no `[ahead N]` before any
  `job execute` or `kernels push`.
- **Corollary: put third-party and repo imports at the TOP, where they fail fast.**
  A late import turns a 30-second failure into a 2.5-hour one.
- **The local CLI dying does NOT kill the job.** On 2026-08-09 `job execute` crashed
  locally with `AssertionError` in `auth.get_md` (`assert current_iam_token`) — an IAM
  token refresh failing in the attached client. The job stayed `EXECUTING` server-side
  and completed normally. Treat a local crash as a lost log stream, not a lost run:
  confirm with `job list -p <PROJECT>` before relaunching, or you will pay twice.
- **Do not pipe `job execute` through `grep`.** It streams; grep block-buffers and
  the launch looks silent. Check with `job list -p <PROJECT>` instead.
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

### Pushing the raw states back to GitHub

Banks run to hundreds of MB (`kaggle_geomcap` 788 MB, `kaggle_h0bank` 705 MB) and a
single push of that size fails over HTTPS:

```
error: RPC failed; curl 55 Send failure: Broken pipe
send-pack: unexpected disconnect while reading sideband packet
```

The commit is safe locally when this happens — only the transfer failed. Fix:

```bash
git config http.postBuffer 524288000   # 500 MB
git config http.version HTTP/1.1       # HTTP/2 is the usual culprit
git push --no-progress
```

Run it backgrounded; 700 MB takes many minutes and will blow a foreground timeout.
Commit the documents in a SEPARATE commit from the raw states where possible, so a
failed data push does not also hold up the ledger.

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


---

## 5. Monitoring and recovery — the practice, after a day of losing time to it

**Use one command, not per-job polling.** `uv run python -m scripts.runs_status`
reads `scratch/RUNS.json` and prints every job on both platforms in one table;
`--pull` additionally downloads each finished DataSphere job and **reads its raw
logs**, flagging the specific failure signatures this project has actually hit.
Keep the registry current — a job missing from it is a job nobody is watching.

**The status field is not the result, and this cost time three separate times in
one day.** All three failures below reported something other than what happened:

| what the platform said | what actually happened | how to tell |
|---|---|---|
| `ERROR`, with a `RemoteDisconnected` in stderr | **host-RAM OOM.** The disconnect had already retried and succeeded two lines later | look for `bash: ... Killed` — SIGKILL, further down than the traceback |
| `SUCCESS`, printing a confident scientific verdict | **all three experimental arms had died of CUDA OOM**; the verdict came from an `if not sig:` branch firing on an empty list | check the arm counts before reading any conclusion |
| `ERROR`, apparently total loss | **fully recoverable** — the declared `outputs:` file was collected anyway | `download-files --with-logs` and open the JSON |

**Rules that follow.**

1. **Write results incrementally, never once at the end.** DataSphere collects the
   declared `outputs:` file even when the job ends in ERROR. Every completed arm
   survived because of a `json.dump(...)` after each one. There is no resume:
   `job attach` works only on a *running* job, `job fork` starts fresh.
2. **A run that concludes must first check it has something to conclude from.**
   Guard every verdict branch on the count of usable measurements and print
   `VOID, NOT NULL` when that count is zero. An empty result set must never reach
   an `if not significant:` branch. *(This is the D95 failure mode; it recurred in
   a script written hours later, so the guard belongs in the template.)*
3. **Two 3.5B fp32 models do not coexist in one T4 process.** fp32 Huginn is
   ~13.5 GB against 14.75 GB of device memory, and `from_config` peaks about
   1.29 GB higher than the resident model (the untied embedding/lm_head copy).
   Put each arm in its **own job** and compare offline; a within-job A/B is not
   worth the OOM risk when the reference arm is already banked.
4. **Diagnose from the bottom of the log upward.** The proximate cause is the last
   thing printed, not the first alarming thing. A retried-and-succeeded warning
   above a fatal line is the most expensive shape of log to read carelessly.
5. **Distinguish the three exit kinds before re-running:** `Killed` (host OOM,
   change where the model is built), `OutOfMemoryError` (GPU OOM, split the job),
   and a Python traceback (real bug). Each has a different fix, and guessing wrong
   costs a full run — it did.

### Workflow observability — where a paused run's work actually lives

`journal.jsonl` records only **completed** agent() calls. A workflow paused or
killed mid-flight can therefore show `result: null` for everything while its
agents have in fact done substantial work — the transcripts are the recovery path,
not the journal.

Recovering a paused workflow's findings, at zero further agent cost:

```bash
D=~/.claude/projects/<proj>/<session>/subagents/workflows/<run_id>
ls -S "$D"/agent-*.jsonl | head        # biggest transcript = most work done
grep -c StructuredOutput "$D"/agent-*.jsonl
```

then parse the `{"findings": [...]}` blob out of the transcript directly. On
2026-08-09 the journal held **0** usable results while the transcripts held **19
substantiated findings**, six of which turned out to be real defects. Check the
transcripts before re-running anything.

**And cap the fan-out.** The same workflow spawned one verifier per finding with no
limit (~24 agents, zero results). Verification is worth keeping — an auditor that
reports a finding without opening the file is the failure mode — but batch it at a
fixed cap (`MAX_VERIFIERS`) so agent count does not scale with how much the
auditors happen to find, and run it on a cheaper model, since checking that a
quoted line exists is mechanical.
