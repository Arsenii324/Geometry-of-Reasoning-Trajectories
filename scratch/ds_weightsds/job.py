"""Build the huginn-0125 weights ONCE into a DataSphere dataset, so no later job downloads them.

WHY. Every job in this project pays 262-282 s fetching 15.65 GB of huginn shards from
HuggingFace before any compute. The startup probe established both halves of the only
CLI-reachable way to remove that phase: a job can WRITE a dataset via `output-datasets:`
(probe D, /job/output-datasets/<name> is a writable 1 GB device), and a later job can
MOUNT it read-only via `datasets:` (probe G, dataset bt1glucd0h22vcddas9t appeared at
/job/datasets/<id> with probe D's payload intact). Nobody built the real one. This does.

LAYOUT, and why it is this way. The weights land at the dataset ROOT -- exactly the layout
probe G verified -- so the mount path is directly loadable:
    AutoModelForCausalLM.from_pretrained("/job/datasets/<id>", trust_remote_code=True)
The pip wheel cache goes in a `wheels/` SUBDIR. Probe D wrote no subdirectory, so
subdirectory survival across the dataset snapshot is not verified; putting the weights at
the root means the deliverable that matters does not depend on that unverified property.

SELF-CHECK BEFORE THE MANIFEST (CLAUDE.md section 5). `snapshot_download(local_dir=...)`
in huggingface_hub < 0.23 populated local_dir with SYMLINKS into ~/.cache/huggingface --
which would put the actual blobs OUTSIDE the dataset and produce a mount full of dangling
links that fails only in the job that tries to use it, weeks later. So: the version is
pinned recent, and every file is then checked to be a real file of the exact byte size
the HF API reported for this revision (fetched locally 2026-08-10, hardcoded below as
ground truth NOT derived from this download). A mismatch sets weights_ok=false in the
MANIFEST rather than being papered over.

EXIT CODE, deliberately 0 EVEN ON FAILURE, which is the opposite of every science kernel
here and needs its reason stated. A non-zero exit puts the job in ERROR, and on ERROR
`download-files` refuses outright (REMOTE_RUNS.md); whether an errored job still COMMITS
its output-dataset is unverified. Throwing away a completed 15 GB download because one
file came up a byte short is the worse failure. So the truth lives in `weights_ok` in the
MANIFEST, not in the job status -- a SUCCESS here means "the job ran", never "the dataset
is good". Read the MANIFEST before mounting it.

PREDICTION, written before the run: all 16 files land as regular files totalling
15,651,519,639 bytes; the c1.4 tier fetches them at HF's rate rather than the 8.3 MB/s
PyPI rate the probe measured, so 10-35 min; the wheel cache adds ~3.5-4 GB, keeping the
total under the 25 Gb the dataset asks for.

The seconds measured here are a CPU-tier number and must never be quoted as a GPU-tier
saving (REMOTE_RUNS.md); what transfers is the mechanism and the BYTES.
"""

import json
import os
import subprocess
import sys
import threading
import time

MODEL_ID = "tomg-group-umd/huginn-0125"
REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"

# Ground truth from https://huggingface.co/api/models/.../tree/<REVISION>, read locally
# before this job was written. This is the external anchor for the self-check: it does not
# come from the download it is checking.
EXPECTED = {
    ".gitattributes": 1566,
    "README.md": 13691,
    "asset1.gif": 3015077,
    "asset2.jpeg": 46921,
    "config.json": 1447,
    "generation_config.json": 178,
    "model-00001-of-00004.safetensors": 4771970936,
    "model-00002-of-00004.safetensors": 4744780096,
    "model-00003-of-00004.safetensors": 4744737616,
    "model-00004-of-00004.safetensors": 1384120448,
    "model.safetensors.index.json": 6188,
    "raven_config_minimal.py": 3913,
    "raven_modeling_minimal.py": 88021,
    "special_tokens_map.json": 435,
    "tokenizer.json": 2726482,
    "tokenizer_config.json": 6624,
}
EXPECTED_TOTAL = 15651519639

# What every kernel in this repo installs, both recipes (REMOTE_RUNS.md "TWO INSTALL
# RECIPES COEXIST"): the model extra and the repo's own dependencies.
WHEEL_SETS = [
    ["torch==2.5.1"],
    ["transformers==4.53.3", "accelerate", "safetensors", "datasets",
     "huggingface_hub", "numpy", "scipy", "scikit-learn", "matplotlib",
     "pandas", "tqdm", "pyyaml"],
]

T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def run(cmd, timeout=3600):
    log("$ %s" % cmd)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    log("  rc=%d  %s" % (p.returncode, out[-1500:]))
    return p.returncode, out


def tree_bytes(root):
    """(n_files, total_bytes) under root, following nothing -- symlinks count as 0."""
    n, tot = 0, 0
    for dirpath, _d, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            n += 1
            tot += os.path.getsize(p)
    return n, tot


def watcher(root, stop):
    """Progress without tqdm's megabytes of carriage returns in the job log."""
    while not stop.is_set():
        stop.wait(60)
        try:
            n, tot = tree_bytes(root)
            log("  ... %d files, %.2f GB on the dataset device" % (n, tot / 1e9))
        except Exception as e:  # noqa: BLE001 -- a watcher must never kill the job
            log("  ... watcher: %r" % e)


def main():
    out_path = os.path.abspath(sys.argv[1])
    ds = os.path.abspath(sys.argv[2])
    res = {"job": "ds_weightsds", "model_id": MODEL_ID, "revision": REVISION,
           "argv": sys.argv, "cwd": os.getcwd(), "dataset_dir": ds,
           "expected_total_bytes": EXPECTED_TOTAL}
    log("start; dataset dir = %s (exists=%s)" % (ds, os.path.isdir(ds)))

    def bank():
        """Write BOTH copies of the manifest after every stage: the collected output and
        the one that lives inside the dataset. Incremental, per REMOTE_RUNS.md section 5."""
        res["elapsed_s"] = round(time.time() - T0, 1)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        try:
            with open(os.path.join(ds, "MANIFEST.json"), "w") as f:
                json.dump(res, f, indent=1)
        except Exception as e:  # noqa: BLE001
            log("could not write MANIFEST into the dataset: %r" % e)

    rc, out = run("df -h %s | tail -2" % ds)
    res["df_before"] = out
    bank()

    # huggingface_hub only. Nothing else is needed to move bytes, and this job must not
    # depend on the torch stack it is caching.
    run("pip install -q --no-cache-dir 'huggingface_hub>=0.30'")
    _, res["hub_version"] = run("python -c \"import huggingface_hub as h; print(h.__version__)\"")

    from huggingface_hub import snapshot_download

    # --- stage 1: the weights, at the dataset ROOT ---
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    stop = threading.Event()
    threading.Thread(target=watcher, args=(ds, stop), daemon=True).start()
    t = time.time()
    try:
        path = snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=ds,
                                 max_workers=8)
        res["snapshot_path"] = path
        res["weights_download_s"] = round(time.time() - t, 1)
        log("snapshot_download returned %s in %.1fs" % (path, res["weights_download_s"]))
    except Exception as e:  # noqa: BLE001
        res["weights_ok"] = False
        res["weights_error"] = repr(e)
        res["weights_download_s"] = round(time.time() - t, 1)
        log("SNAPSHOT FAILED: %r" % e)
        stop.set()
        bank()
        return 0          # see EXIT CODE note in the module docstring
    stop.set()

    # --- the self-check, before anything downstream is told the dataset is good ---
    files, problems = {}, []
    for rel, want in EXPECTED.items():
        p = os.path.join(ds, rel)
        if not os.path.exists(p):
            problems.append("%s MISSING" % rel)
            continue
        if os.path.islink(p):
            problems.append("%s is a SYMLINK -> %s (blob is outside the dataset)"
                            % (rel, os.readlink(p)))
            continue
        got = os.path.getsize(p)
        files[rel] = got
        if got != want:
            problems.append("%s is %d bytes, expected %d" % (rel, got, want))
    n_all, tot_all = tree_bytes(ds)
    res["weights_files"] = files
    res["weights_bytes"] = sum(files.values())
    res["weights_problems"] = problems
    res["weights_ok"] = (not problems) and sum(files.values()) == EXPECTED_TOTAL
    res["dataset_files_total"] = n_all
    res["dataset_bytes_total"] = tot_all
    log("SELF-CHECK: %d/%d files, %d bytes vs expected %d -> %s"
        % (len(files), len(EXPECTED), res["weights_bytes"], EXPECTED_TOTAL,
           "OK" if res["weights_ok"] else "FAILED: " + "; ".join(problems)))
    bank()
    if not res["weights_ok"]:
        log("DATASET NOT USABLE -- weights_ok=false. Skipping the wheel cache; read the "
            "MANIFEST, do not mount this dataset for science.")
        return 0          # see EXIT CODE note in the module docstring

    # --- stage 2: the pip wheel cache, in a subdir, and it may not endanger stage 1 ---
    # pip resolves on platform tags, not GPU presence: the probe measured c1.4 fetching
    # 3005.8 MB for torch==2.5.1 against the GPU tier's 3006.0 MB, i.e. the same wheel.
    wheels = os.path.join(ds, "wheels")
    try:
        os.makedirs(wheels, exist_ok=True)
        _, res["pip_tags"] = run("python -c \"import sysconfig,platform,sys;"
                                 "print(sys.version, platform.machine(), "
                                 "sysconfig.get_platform())\"")
        res["wheel_sets"] = []
        for spec in WHEEL_SETS:
            t = time.time()
            rc, out = run("pip download --no-cache-dir --dest %s %s"
                          % (wheels, " ".join("'%s'" % s for s in spec)))
            res["wheel_sets"].append({"spec": spec, "rc": rc,
                                      "seconds": round(time.time() - t, 1),
                                      "tail": out[-800:]})
            bank()
        wn, wb = tree_bytes(wheels)
        res["wheels_files"] = sorted(os.listdir(wheels))
        res["wheels_count"] = wn
        res["wheels_bytes"] = wb
        res["wheels_ok"] = all(w["rc"] == 0 for w in res["wheel_sets"]) and wn > 0
        log("wheel cache: %d files, %.2f GB, ok=%s" % (wn, wb / 1e9, res["wheels_ok"]))
    except Exception as e:  # noqa: BLE001
        res["wheels_ok"] = False
        res["wheels_error"] = repr(e)
        log("wheel cache failed (weights are already banked and unaffected): %r" % e)

    n_all, tot_all = tree_bytes(ds)
    res["dataset_files_total"] = n_all
    res["dataset_bytes_total"] = tot_all
    _, res["df_after"] = run("df -h %s | tail -2" % ds)
    _, res["listing"] = run("ls -la %s" % ds)
    bank()
    log("DONE: %d files, %.2f GB total, weights_ok=%s wheels_ok=%s, %.1fs"
        % (n_all, tot_all / 1e9, res["weights_ok"], res.get("wheels_ok"),
           time.time() - T0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
