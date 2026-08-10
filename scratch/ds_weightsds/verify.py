"""Verify the huginn weights dataset MOUNTS and LOADS, before anything is told to use it.

The build job's self-check proved the bytes were correct in /job/output-datasets/... .
It did NOT prove they survive the snapshot into a dataset, that the `wheels/` SUBDIR
survives (probe D wrote no subdirectory, so that is unverified), or -- the number that
decides whether any of this was worth doing -- how long MOUNTING 21.56 GB costs. A mount
that takes as long as the 262-282 s download it replaces is not a saving.

Three things measured here, each of which a later kernel's correctness depends on:

  1. MOUNT COST. The job proto's started_at against this script's first log line. The
     build job, whose output-dataset was an empty 25 GB device, took 8 s. Anything near
     that means the mount is effectively free and the saving is the full download.
  2. THE BYTES SURVIVED. Every file re-measured on the READ-ONLY mount, against the
     manifest the build wrote AND against the total from the HF API, which is external to
     both. Then a real shard is opened with safetensors and a tensor actually read, since
     a correct file size proves nothing about a corrupted transfer.
  3. IT LOADS FROM THE MOUNT. `trust_remote_code=True` makes transformers import
     raven_modeling_minimal.py out of the model directory -- and this directory is
     READ-ONLY, which is the plausible way this whole scheme fails. Loading the config and
     tokenizer exercises that import path without needing the 14 GB the full fp32 model
     would want on a CPU pod.

PREDICTION: mount under ~30 s; 16 files at 15,651,519,639 bytes; wheels/ present with 94
files; the offline install from the cached wheels succeeds with no network; AutoConfig and
AutoTokenizer both load from the mount path.
"""

import json
import os
import subprocess
import sys
import time

EXPECTED_TOTAL = 15651519639          # HF API, revision bb6621b6 -- external to the build
N_EXPECTED = 16
INSTALL = ("torch==2.5.1 transformers==4.53.3 accelerate safetensors datasets "
           "numpy scipy scikit-learn matplotlib pandas tqdm pyyaml")

T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def run(cmd, timeout=1800):
    log("$ %s" % cmd)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    log("  rc=%d  %s" % (p.returncode, out[-1200:]))
    return p.returncode, out


def main():
    out_path = os.path.abspath(sys.argv[1])
    mnt = os.path.abspath(sys.argv[2])
    res = {"job": "ds_weightsds_verify", "mount": mnt, "argv": sys.argv,
           "script_start_epoch": T0}
    log("MOUNT-COST CLOCK: this line's wall-clock, minus the job proto's started_at, "
        "is the cost of mounting 21.56 GB. mount=%s exists=%s" % (mnt, os.path.isdir(mnt)))

    def bank():
        res["elapsed_s"] = round(time.time() - T0, 1)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)

    _, res["free"] = run("free -g")
    _, res["df"] = run("df -h / %s" % mnt)
    _, res["listing"] = run("ls -la %s" % mnt)
    bank()

    # --- 2. the bytes ---
    sizes, links = {}, []
    for dirpath, _d, files in os.walk(mnt):
        for f in files:
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, mnt)
            if os.path.islink(p):
                links.append(rel)
                continue
            sizes[rel] = os.path.getsize(p)
    top = {k: v for k, v in sizes.items() if os.sep not in k}
    weight_files = {k: v for k, v in top.items() if k != "MANIFEST.json"}
    res["symlinks"] = links
    res["n_top_level"] = len(top)
    res["weights_bytes_on_mount"] = sum(weight_files.values())
    res["weights_total_matches_hf_api"] = (
        res["weights_bytes_on_mount"] == EXPECTED_TOTAL
        and len(weight_files) == N_EXPECTED)
    res["wheels_present"] = os.path.isdir(os.path.join(mnt, "wheels"))
    res["wheels_count"] = (len(os.listdir(os.path.join(mnt, "wheels")))
                           if res["wheels_present"] else 0)
    log("bytes on mount %d vs HF API %d ; %d top-level files ; wheels/ present=%s (%d) ; "
        "symlinks=%d" % (res["weights_bytes_on_mount"], EXPECTED_TOTAL, len(top),
                         res["wheels_present"], res["wheels_count"], len(links)))

    # cross-check against what the BUILD said it wrote -- catches a partial snapshot
    try:
        man = json.load(open(os.path.join(mnt, "MANIFEST.json")))
        res["manifest_weights_ok"] = man.get("weights_ok")
        res["manifest_agrees"] = (man.get("weights_files") == weight_files)
        log("MANIFEST: weights_ok=%s, per-file sizes agree with the mount: %s"
            % (res["manifest_weights_ok"], res["manifest_agrees"]))
    except Exception as e:  # noqa: BLE001
        res["manifest_error"] = repr(e)
        log("could not read MANIFEST.json: %r" % e)
    bank()

    # --- the install, from the cached wheels, with the index switched OFF ---
    t = time.time()
    rc, _ = run("pip install --no-index --find-links=%s/wheels %s" % (mnt, INSTALL))
    res["offline_install_rc"] = rc
    res["offline_install_s"] = round(time.time() - t, 1)
    res["offline_install_ok"] = (rc == 0)
    log("OFFLINE install from the mounted wheel cache: rc=%d in %.1fs"
        % (rc, res["offline_install_s"]))
    if rc != 0:
        t = time.time()
        rc2, _ = run("pip install %s" % INSTALL)
        res["online_fallback_rc"] = rc2
        res["online_fallback_s"] = round(time.time() - t, 1)
    bank()

    # --- 3. does it LOAD from the read-only mount ---
    try:
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig, AutoTokenizer

        t = time.time()
        cfg = AutoConfig.from_pretrained(mnt, trust_remote_code=True)
        res["config_load_s"] = round(time.time() - t, 1)
        res["config_class"] = type(cfg).__name__
        res["config_n_embd"] = getattr(cfg, "n_embd", None)
        res["config_mean_recurrence"] = getattr(cfg, "mean_recurrence", None)
        log("AutoConfig from the mount -> %s n_embd=%s mean_recurrence=%s (%.1fs) "
            "-- remote code imported out of a READ-ONLY dir"
            % (res["config_class"], res["config_n_embd"],
               res["config_mean_recurrence"], res["config_load_s"]))

        t = time.time()
        tok = AutoTokenizer.from_pretrained(mnt)
        ids = tok("The capital of France is", add_special_tokens=False).input_ids
        res["tokenizer_load_s"] = round(time.time() - t, 1)
        res["tokenizer_class"] = type(tok).__name__
        res["tokenizer_vocab"] = int(tok.vocab_size)
        res["probe_ids"] = list(ids)
        log("AutoTokenizer -> %s vocab=%d, 'The capital of France is' -> %r"
            % (res["tokenizer_class"], res["tokenizer_vocab"], ids))

        # a real tensor, not just a correct file size
        shard = os.path.join(mnt, "model-00001-of-00004.safetensors")
        with safe_open(shard, framework="pt") as f:
            keys = list(f.keys())
            k0 = keys[0]
            v = f.get_tensor(k0)
        res["shard_n_tensors"] = len(keys)
        res["shard_first_key"] = k0
        res["shard_first_shape"] = list(v.shape)
        res["shard_first_dtype"] = str(v.dtype)
        res["shard_first_isfinite"] = bool(torch.isfinite(v).all())
        res["shard_first_absmean"] = float(v.abs().float().mean())
        log("shard 1: %d tensors; %s %s %s finite=%s absmean=%.6f"
            % (len(keys), k0, res["shard_first_shape"], res["shard_first_dtype"],
               res["shard_first_isfinite"], res["shard_first_absmean"]))
        res["load_ok"] = bool(res["shard_first_isfinite"]) and res["config_n_embd"] == 5280
    except Exception as e:  # noqa: BLE001
        import traceback
        res["load_ok"] = False
        res["load_error"] = traceback.format_exc()[-2000:]
        log("LOAD FAILED:\n%s" % res["load_error"])

    res["verdict_ok"] = bool(res.get("weights_total_matches_hf_api")
                             and res.get("wheels_present")
                             and res.get("offline_install_ok")
                             and res.get("load_ok"))
    bank()
    log("VERDICT ok=%s (bytes=%s wheels=%s offline_install=%s load=%s) in %.1fs"
        % (res["verdict_ok"], res.get("weights_total_matches_hf_api"),
           res.get("wheels_present"), res.get("offline_install_ok"),
           res.get("load_ok"), time.time() - T0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
