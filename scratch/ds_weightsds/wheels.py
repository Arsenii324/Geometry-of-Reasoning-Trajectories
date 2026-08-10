"""Build the pip wheel cache as its OWN dataset, and measure WHEN the snapshot is taken.

WHY A SECOND DATASET. The weights build (job bt1e9ld08akjoe25q22j) wrote 21.56 GB and its
own end-of-job `df` and file walk confirmed all of it was on the device -- 94 wheels,
5.91 GB, both pip downloads rc=0. The MOUNT of the resulting dataset (bt102r0j5cb8r6r6nb36)
shows 15 GB, an EMPTY wheels/ directory, and a MANIFEST.json of 1347 bytes: the version
written at t=1010 s, not the 3438-byte one written at t=1165 s. So DataSphere snapshots an
`output-datasets:` volume at some moment BEFORE the job ends -- here ~2m45s before -- and
everything written after that moment is silently discarded. The job still reports SUCCESS.

That is the trap, and it is worth more than the wheels: a build job that writes its
manifest last loses exactly the file a later job would check.

SO THIS JOB DOES TWO THINGS AT ONCE.
  1. Delivers the wheel cache, writing it EARLY and then idling, on the hypothesis that
     what survives is what was written sufficiently long before the end.
  2. MEASURES the cut-off, instead of guessing at it: a marker file every 30 s from t=0
     to exit, each followed by a sync. Mounting this dataset and reading the highest
     surviving marker gives the cut-off in seconds before job end, as a number.

PREDICTION: the wheels (all written before t~150 s) survive; the markers survive up to
somewhere around 150-200 s before exit and stop; the last few do not.
"""

import json
import os
import subprocess
import sys
import time

INSTALL = [
    ["torch==2.5.1"],
    ["transformers==4.53.3", "accelerate", "safetensors", "datasets",
     "huggingface_hub", "numpy", "scipy", "scikit-learn", "matplotlib",
     "pandas", "tqdm", "pyyaml"],
]
IDLE_UNTIL_S = 660          # keep marking until here, then exit
MARK_EVERY_S = 30

T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def run(cmd, timeout=1800):
    log("$ %s" % cmd)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    log("  rc=%d  %s" % (p.returncode, out[-600:]))
    return p.returncode, out


def mark(ds, t):
    """A marker whose NAME carries the second it was written at. The highest one that
    survives into the mounted dataset is the measurement."""
    p = os.path.join(ds, "marker_t%04d.txt" % int(t))
    with open(p, "w") as f:
        f.write("t=%.2f epoch=%.2f\n" % (t, time.time()))
    os.sync()
    return p


def main():
    out_path = os.path.abspath(sys.argv[1])
    ds = os.path.abspath(sys.argv[2])
    res = {"job": "ds_weightsds_wheels", "dataset_dir": ds, "argv": sys.argv,
           "script_start_epoch": T0, "idle_until_s": IDLE_UNTIL_S,
           "mark_every_s": MARK_EVERY_S, "markers": []}
    log("start; %s exists=%s" % (ds, os.path.isdir(ds)))

    def bank():
        res["elapsed_s"] = round(time.time() - T0, 1)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)

    res["markers"].append(os.path.basename(mark(ds, 0)))
    _, res["df_before"] = run("df -h %s | tail -1" % ds)
    bank()

    wheels = os.path.join(ds, "wheels")
    os.makedirs(wheels, exist_ok=True)
    res["sets"] = []
    for spec in INSTALL:
        t = time.time()
        rc, out = run("pip download --no-cache-dir --dest %s %s"
                      % (wheels, " ".join("'%s'" % s for s in spec)))
        res["sets"].append({"spec": spec, "rc": rc, "seconds": round(time.time() - t, 1)})
        bank()
    names = sorted(os.listdir(wheels))
    res["wheels_count"] = len(names)
    res["wheels_bytes"] = sum(os.path.getsize(os.path.join(wheels, n)) for n in names)
    res["wheels_ok"] = all(s["rc"] == 0 for s in res["sets"]) and len(names) > 0
    log("wheel cache: %d files, %.2f GB, ok=%s"
        % (res["wheels_count"], res["wheels_bytes"] / 1e9, res["wheels_ok"]))

    # The manifest goes in EARLY -- writing it last is what lost it the first time.
    with open(os.path.join(ds, "MANIFEST.json"), "w") as f:
        json.dump({"kind": "pip wheel cache", "built_by": "ds_weightsds/wheels.py",
                   "wheels_count": res["wheels_count"], "wheels_bytes": res["wheels_bytes"],
                   "wheels": names, "sets": res["sets"],
                   "written_at_s": round(time.time() - T0, 1)}, f, indent=1)
    os.sync()
    log("MANIFEST written at t=%.1fs and synced" % (time.time() - T0))
    bank()

    # --- the measurement: keep marking, right up to the exit ---
    nxt = MARK_EVERY_S * (int((time.time() - T0) // MARK_EVERY_S) + 1)
    while True:
        now = time.time() - T0
        if now >= IDLE_UNTIL_S:
            break
        if now >= nxt:
            res["markers"].append(os.path.basename(mark(ds, nxt)))
            log("marker t=%d (%d so far)" % (nxt, len(res["markers"])))
            nxt += MARK_EVERY_S
            bank()
        time.sleep(2)
    res["markers"].append(os.path.basename(mark(ds, IDLE_UNTIL_S)))

    _, res["df_after"] = run("df -h %s | tail -1" % ds)
    _, res["listing"] = run("ls -la %s | head -30" % ds)
    res["last_marker_s"] = IDLE_UNTIL_S
    res["exit_at_s"] = round(time.time() - T0, 1)
    bank()
    log("DONE: %d wheels, %d markers, last marker t=%d, exiting at t=%.1fs"
        % (res["wheels_count"], len(res["markers"]), IDLE_UNTIL_S, res["exit_at_s"]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
