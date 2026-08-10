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
- **DataSphere DOES run jobs in parallel; a refusal is transient capacity.**
  `INVALID_ARGUMENT: Instance types gt4.1 are not available for your community`
  looks like an entitlement error and is not one: **retry it.** On 2026-08-09 the
  identical launch was refused twice (gt4.1 and g4i.1) and then succeeded minutes
  later with the other job still EXECUTING. The project's own job timestamps show
  **6 overlapping pairs**, including a three-way overlap at 12:26-13:08. A previous
  revision of this file concluded a structural concurrency cap from ONE refusal --
  wrong, and worth keeping as the cautionary example: a single failed attempt is
  evidence about that attempt, not about the platform.
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
## Cutting the ~650 s per-job startup (measured 2026-08-10, on the CHEAPEST tier)

Every GPU job pays ~262-282 s downloading the model and ~390 s on pip before any compute.
A probe agent tested the options on `c1.4` (no GPU spend) against the CLI's own config
parser, not the docs:

| mechanism | supported | note |
|---|---|---|
| `env.docker: {image: ...}` | **yes** | works on GPU tiers here — job `bt1hd3oqb17690amgolg` ran `nvidia/cuda:12.2.2` on g1.1 |
| `datasets: [<id>: VAR]` | **yes** | read-only mount at `/job/datasets/<id>`; full create→mount round trip verified |
| `flags: [attach-project-disk]` | **yes** | writable, **persists across jobs**, but only **10.74 GB** — too small for the 15.65 GB of weights |
| cached/named python env | **NO — does not exist** | a fresh venv per job (`/job/.job_python_venv_*` differs every run); the docs' caching claim is about INPUT FILES |

**The validity check that made the cheap tier usable:** pip resolves on platform tags, not
GPU presence, so `c1.4` downloaded **3005.8 MB** for `torch==2.5.1` against the GPU job's
**3006.0 MB** — the bytes transfer, so mechanism and volume are testable there. The SECONDS
do not: PyPI measured 8.3 MB/s from `c1.4` against ~55 MB/s on the GPU pod. Never quote a
cheap-tier duration as a GPU-tier saving.

**Recommended, unambiguous:** put the weights in a dataset and mount them — removes the whole
262-282 s phase, and `snapshot_download` into an `output-datasets:` entry can be built by a
`c1.4` job with no GPU. The dataset ID is **not printed by `job get`**; it lives only in the
Job proto's `output_datasets[].id`.

**A genuine choice, not made here (CLAUDE.md section 8):** the pip phase. Docker Hub's
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` carries the identical CUDA stack with zero pip,
but is 3.34 GB and measured a **204-219 s cold pull**; a wheel cache on the project disk still
pays **155.4 s** to install. Both beat ~390 s, neither is free, and a custom image on Yandex
Container Registry would likely beat both (in-cloud pull, and it could bake in `transformers`,
which the pytorch image lacks) — untested, and creating a registry is an account change.

*Left behind by the probe: dataset `bt1glucd0h22vcddas9t` (1 GB marker) and two small files on
the project disk, both safe to delete.*

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

## Post-mortem on a failed job, and the two install recipes (2026-08-10)

**`datasphere project job attach --id <job>` is the only way to see why a finished
job failed.** `download-files` refuses outright on `ERROR` status ("job ... was
completed with error (5). Not all files can be downloaded" -> "no files to
download"), and the CLI's local follower log under
`/var/folders/.../T/datasphere/job_*` is empty once the launching `timeout` has
killed the follower. `attach` re-streams the job's stdout/stderr from the server,
including the full Python traceback, and works long after the job ended. It exits
non-zero with `ProgramError: Program returned code 1`, which is expected -- read the
traceback above it.

**This mattered.** A21 failed 12 minutes in. From timing alone the obvious diagnosis
was a CUDA OOM: Huginn in float32 is ~14 GB on a 16 GB T4, and A21 is the first
kernel needing autograd rather than pure forwards. `attach` showed the actual cause
in one line -- `ModuleNotFoundError: No module named 'scipy'` -- which is a
30-second fix rather than a redesign. **Retrieve the log before theorising.**

**TWO INSTALL RECIPES COEXIST AND THEY ARE NOT INTERCHANGEABLE.**

| recipe | what it installs | used by |
|---|---|---|
| `pip install -e .[model]` | the repo **and its dependencies** -- numpy, scipy, scikit-learn, matplotlib, pandas, tqdm, pyyaml, plus the `model` extra | `ds_nth`, `ds_einterp`, `ds_estream` |
| `pip install transformers==4.53.3 accelerate safetensors` | only what is named; **no repo dependencies at all** | `ds_gmres`, `ds_embsep` |

The lean recipe is legitimate and faster -- it skips sklearn/matplotlib/pandas/datasets.
What is not legitimate is combining it with an import of a repo dependency. Because
this project deliberately places third-party imports LATE (inside `main()`, after the
clone), the failure surfaces at the moment that line executes, which is after every
fixed cost has been paid. Kernels using the lean recipe must name every package they
import.

`scripts/preflight.py` check 4 now enforces this: it cross-checks each kernel's
imports against its own pip lines, skips `try:`-guarded optional imports, and was
verified to fail the real pre-fix file and to produce no false positives on any of
the 24 kernels in `scratch/`.

## The weights are now a dataset: stop downloading them (built and verified 2026-08-10)

**The 262-282 s HuggingFace download is gone. Mount this instead.** Built by
`scratch/ds_weightsds/` on `c1.4`, no GPU.

| | |
|---|---|
| **dataset id** | **`bt102r0j5cb8r6r6nb36`** (name `huginn-0125-weights`, 25 Gb) |
| contents | `tomg-group-umd/huginn-0125` at revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`, snapshot at the dataset ROOT, plus `MANIFEST.json` |
| build | job `bt1e9ld08akjoe25q22j`, **19 m 47 s** wall (20:40:13→20:59:59 UTC); the download itself was 1005.3 s at ~15.6 MB/s on `c1.4` |
| verified | job `bt1uj1rm0fhnvm82srhd` mounted it: 16 files, **15,651,519,639 bytes**, zero symlinks, byte-exact against the HF API's own per-file sizes |
| **mount cost** | **7.7 s** — against 8.1 s for the *empty* output-dataset device on the build job, and 9.2 s for two datasets at once. Mounting is ~constant in dataset size. |

**Recovering the id of a dataset a job just built** — still not printed by `job get`, and
`job list` does not carry it either. One command, and it works on a finished job:

```bash
GRPC_DNS_RESOLVER=native ~/.local/pipx/venvs/datasphere/bin/python \
  scratch/ds_startup_probe/dump_job.py <JOB_ID>      # -> OUTPUT DATASET id=... name=... size_gb=...
```

The mount is read-only at `/job/datasets/<id>`, and the path also arrives in `argv`.

```yaml
datasets:
  - bt102r0j5cb8r6r6nb36: HUGINN
cmd: python job.py ${RESULT} ${HUGINN}
```

Then pass the mount path where `MODEL_ID` used to go. **Do not pass `revision=`** — the
revision is baked into the dataset, and there is nothing to resolve against the Hub.

```python
mnt = os.path.abspath(sys.argv[2])                      # /job/datasets/bt102r0j5cb8r6r6nb36
tok = AutoTokenizer.from_pretrained(mnt)
cfg = AutoConfig.from_pretrained(mnt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    mnt, config=cfg, trust_remote_code=True,
    torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda").eval()
```

`trust_remote_code=True` makes transformers import `raven_modeling_minimal.py` out of the
model directory, and that directory is **read-only** — the plausible way this whole scheme
fails. It does not: transformers copies the module into its own writable modules cache.
Measured on the mount, `AutoConfig` returned `RavenConfig` with `n_embd=5280`,
`mean_recurrence=32` in 0.2 s, `AutoTokenizer` gave `PreTrainedTokenizerFast` with
vocab 65536, and `safetensors.safe_open` on shard 1 read 23 finite tensors. The full
`from_pretrained` above was NOT run (a CPU pod will not hold 14 GB of fp32); everything
short of materialising the weights was.

Cheap verification from inside a later job, before trusting the mount:

```python
man = json.load(open(os.path.join(mnt, "MANIFEST.json")))   # weights_ok, per-file sizes
assert man["weights_ok"] and sum(man["weights_files"].values()) == 15651519639
```

### The trap that ate 5.91 GB: an output-dataset snapshot does NOT include your page cache

The build job also wrote a 94-file, 5.91 GB pip wheel cache into `wheels/`. Its own
end-of-job `df` said **21G used** and its file walk counted **146 files / 21,561,595,604
bytes**, both `pip download`s returned rc=0, and the job reported **SUCCESS**. The mounted
dataset contains **15G**, an **empty `wheels/` directory**, and a `MANIFEST.json` of 1347
bytes — the version written at t=1010 s, not the 3438-byte one written at t=1165 s.
Everything from the last ~2m45s was silently discarded.

Two explanations fit that, and they imply different rules, so it was measured rather than
guessed. A second job (`bt1pk4fk6lvjl0ogc28n`) wrote a marker file every 30 s until it
exited at t=661.4 s, calling `os.sync()` after each. **All ten markers survived, including
`marker_t0660` written ~1 s before exit.** So the snapshot is not taken early; what is lost
is data still sitting in the guest's page cache when the volume is read. The unplanned
corroboration is in that job's own timing: its `os.sync()` **blocked for ~257 s**
(t=154.5→t=411.3) flushing the 5.91 GB that pip had "finished" writing minutes earlier.

**Rule: call `os.sync()` after the last write to an `output-datasets:` volume, and let it
return before the job exits.** It can take minutes for multi-GB payloads, and that time is
the job's, not the platform's. Write the manifest EARLY, not last — writing it last is
exactly what lost it here. And a build job's own `df` proves nothing: only mounting the
finished dataset does.

### The wheel cache exists, and on the evidence it is not worth mounting

Rebuilt with the sync fix as its own dataset — **`bt1k1pq7maoh30b29t1l`**
(`huginn-pip-wheels`, 12 Gb): 94 wheels, 5.91 GB, for `torch==2.5.1`,
`transformers==4.53.3`, accelerate, safetensors, datasets and the repo's own dependencies.

```yaml
datasets:
  - bt1k1pq7maoh30b29t1l: WHEELS
# pip install --no-index --find-links=${WHEELS}/wheels torch==2.5.1 transformers==4.53.3 ...
```

It works — `--no-index` installed the whole set with no network. **But it was slower than
just using PyPI.** On `c1.4`, the offline install from the mount took **340.9 s**
(job `bt1ms84pjdlt4q5dn17e`) against **248.0 s** for the ordinary online install of the
identical package list (job `bt1uj1rm0fhnvm82srhd`). Reading 5.91 GB off the dataset mount
costs more than fetching it, and the unpack is CPU-bound either way.

Caveats, both directions: these are one measurement each on different pods, not a
controlled A/B, and both are **cheap-tier** numbers — the rule against quoting a `c1.4`
duration as a GPU-tier saving applies here too, since PyPI runs ~8.3 MB/s from `c1.4`
against ~55 MB/s on a GPU pod, and the mount's read throughput on a GPU tier is untested.
So: **mount the weights (a clear ~270 s saving for ~8 s), and do not bother with the wheels
unless someone measures them on the tier that matters.** The dataset is there if they do.
