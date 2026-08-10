"""Mount BOTH datasets and settle what a kernel may actually rely on.

THE DISCRIMINATING TEST. The weights build lost 5.91 GB of wheels written in its last
2m45s, and two explanations fit that equally well:

  H_time  DataSphere snapshots the output-dataset volume some fixed time BEFORE the job
          ends, so late writes are never in scope.
  H_sync  The snapshot reads the BLOCK DEVICE, so anything still sitting in the guest's
          page cache is lost -- and 5.91 GB takes minutes to write back. Supporting
          evidence, unplanned: the wheels job's `os.sync()` BLOCKED for ~257 s (t=154.5 ->
          t=411.3) flushing exactly those wheels, long after pip had returned.

The wheels job wrote a marker every 30 s to t=660 and exited at t=661.4, calling os.sync()
after EACH one. The two hypotheses now disagree, sharply:
  under H_time  the last markers are missing -- nothing after roughly t=500 survives;
  under H_sync  ALL TEN survive, including t=660, written ~1 s before the job exited.
Whichever way it falls, the operational rule that comes out of it is different, so this is
worth one CPU job.

Also retried here: the offline `pip install --no-index` that failed on the empty wheels/
directory, and a load of the model config from the weights mount using the packages that
install produced -- i.e. the exact sequence a real kernel would run.

PREDICTION: all 10 markers present (H_sync); 94 wheels, 5.91 GB; the offline install
succeeds with no network; RavenConfig loads from the weights mount.
"""

import json
import os
import re
import subprocess
import sys
import time

INSTALL = ("torch==2.5.1 transformers==4.53.3 accelerate safetensors datasets "
           "numpy scipy scikit-learn matplotlib pandas tqdm pyyaml")
WEIGHTS_TOTAL = 15651519639
LAST_MARKER_S = 660          # what the wheels job wrote last, exiting at t=661.4

T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def run(cmd, timeout=1800):
    log("$ %s" % cmd)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    log("  rc=%d  %s" % (p.returncode, out[-900:]))
    return p.returncode, out


def main():
    out_path = os.path.abspath(sys.argv[1])
    wmnt, pmnt = os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
    res = {"job": "ds_weightsds_verify_both", "weights_mount": wmnt,
           "wheels_mount": pmnt, "script_start_epoch": T0}
    log("MOUNT CLOCK -- this line minus the proto's started_at is the cost of mounting "
        "BOTH datasets. weights=%s wheels=%s" % (os.path.isdir(wmnt), os.path.isdir(pmnt)))

    def bank():
        res["elapsed_s"] = round(time.time() - T0, 1)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)

    _, res["df"] = run("df -h %s %s | tail -3" % (wmnt, pmnt))
    _, res["wheels_listing"] = run("ls -la %s | head -20" % pmnt)
    bank()

    # --- the discriminating test ---
    marks = sorted(int(m.group(1)) for m in
                   (re.match(r"marker_t(\d+)\.txt$", n) for n in os.listdir(pmnt)) if m)
    res["markers_present"] = marks
    res["n_markers"] = len(marks)
    res["last_marker_present_s"] = max(marks) if marks else None
    res["all_markers_survived"] = (len(marks) == 10 and max(marks) == LAST_MARKER_S)
    res["verdict_mechanism"] = ("H_sync: page-cache writeback, not snapshot timing"
                                if res["all_markers_survived"] else
                                "H_time or partial: snapshot precedes job end")
    log("MARKERS present: %r ; last=%s of %s written ; -> %s"
        % (marks, res["last_marker_present_s"], LAST_MARKER_S, res["verdict_mechanism"]))

    wdir = os.path.join(pmnt, "wheels")
    names = sorted(os.listdir(wdir)) if os.path.isdir(wdir) else []
    res["wheels_count"] = len(names)
    res["wheels_bytes"] = sum(os.path.getsize(os.path.join(wdir, n)) for n in names)
    log("wheels/: %d files, %.2f GB" % (len(names), res["wheels_bytes"] / 1e9))

    wf = {n: os.path.getsize(os.path.join(wmnt, n)) for n in os.listdir(wmnt)
          if os.path.isfile(os.path.join(wmnt, n)) and n != "MANIFEST.json"}
    res["weights_bytes_on_mount"] = sum(wf.values())
    res["weights_ok"] = (res["weights_bytes_on_mount"] == WEIGHTS_TOTAL and len(wf) == 16)
    log("weights mount: %d files, %d bytes, ok=%s"
        % (len(wf), res["weights_bytes_on_mount"], res["weights_ok"]))
    bank()

    # --- the offline install, with the index switched OFF ---
    t = time.time()
    rc, _ = run("pip install --no-index --find-links=%s %s" % (wdir, INSTALL))
    res["offline_install_rc"] = rc
    res["offline_install_s"] = round(time.time() - t, 1)
    res["offline_install_ok"] = (rc == 0)
    log("OFFLINE install from the mounted wheel cache: rc=%d in %.1fs (no network used)"
        % (rc, res["offline_install_s"]))
    bank()

    try:
        from transformers import AutoConfig, AutoTokenizer
        t = time.time()
        cfg = AutoConfig.from_pretrained(wmnt, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(wmnt)
        res["load_s"] = round(time.time() - t, 1)
        res["config_class"] = type(cfg).__name__
        res["config_n_embd"] = getattr(cfg, "n_embd", None)
        res["tokenizer_vocab"] = int(tok.vocab_size)
        res["load_ok"] = (res["config_n_embd"] == 5280 and res["tokenizer_vocab"] == 65536)
        log("loaded from the mount with the OFFLINE-installed packages: %s n_embd=%s "
            "vocab=%s in %.1fs" % (res["config_class"], res["config_n_embd"],
                                   res["tokenizer_vocab"], res["load_s"]))
    except Exception:  # noqa: BLE001
        import traceback
        res["load_ok"] = False
        res["load_error"] = traceback.format_exc()[-1500:]
        log("LOAD FAILED:\n%s" % res["load_error"])

    res["verdict_ok"] = bool(res.get("weights_ok") and res.get("offline_install_ok")
                             and res.get("load_ok") and res["wheels_count"] > 0)
    res["total_setup_s"] = round(time.time() - T0, 1)
    bank()
    log("VERDICT ok=%s | markers=%s | weights=%s wheels=%d offline_install=%s load=%s "
        "| whole setup %.1fs against 262-282 s of download plus ~390 s of pip"
        % (res["verdict_ok"], res["all_markers_survived"], res.get("weights_ok"),
           res["wheels_count"], res.get("offline_install_ok"), res.get("load_ok"),
           res["total_setup_s"]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
