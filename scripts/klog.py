"""Extract clean stdout from a Kaggle kernel log, and pull JSON records out of it.

Kaggle wraps every print in a JSON-lines envelope:

    {"stream_name":"stdout","time":12.3,"data":"  rho = 0.88\\n"}

so a 19 KB log carries ~3 KB of signal. This parser was rewritten inline at least
three times during this project, once with a regex that missed multi-line records.
It is also the reason an adversarial review burned ~30M cache-read tokens: every
agent re-parsed the same wrapped logs itself.

Records emitted by kernels as one-line JSON (the `{"n_ops": ...}` convention) are
recoverable with `--json`, which is how D55 was derived from a log written a week
earlier.

Usage
    python -m scripts.klog <logfile>              clean stdout
    python -m scripts.klog <logfile> --tail 40    last 40 lines
    python -m scripts.klog <logfile> --json       every embedded JSON object
    python -m scripts.klog <logfile> --err        stderr instead
"""

from __future__ import annotations

import json
import re
import sys

_ENVELOPE = re.compile(
    r'"stream_name":"(?P<stream>stdout|stderr)","time":[\d.]+,"data":(?P<data>"(?:[^"\\]|\\.)*")',
    re.S,
)


def text(path: str, stream: str = "stdout") -> str:
    """Concatenated payload of every envelope on `stream`."""
    raw = open(path, encoding="utf-8", errors="replace").read()
    return "".join(json.loads(m.group("data"))
                   for m in _ENVELOPE.finditer(raw)
                   if m.group("stream") == stream)


def records(path: str, key: str | None = None) -> list[dict]:
    """Every embedded JSON object in the stdout stream.

    Uses `raw_decode` from each `{` rather than a regex, so objects spanning many
    lines and containing nested arrays are recovered intact — the failure mode of
    the ad-hoc versions.
    """
    body = text(path)
    dec = json.JSONDecoder()
    out: list[dict] = []
    for m in re.finditer(r"\{", body):
        try:
            obj, _ = dec.raw_decode(body[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and (key is None or key in obj):
            out.append(obj)
    # drop objects fully contained in an earlier one
    seen: set[str] = set()
    uniq = []
    for o in out:
        s = json.dumps(o, sort_keys=True)
        if s not in seen:
            seen.add(s)
            uniq.append(o)
    return uniq


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    path = args[0]
    stream = "stderr" if "--err" in argv else "stdout"
    if "--json" in argv:
        recs = records(path)
        print(f"# {len(recs)} embedded JSON objects")
        for r in recs:
            print(json.dumps(r))
        return 0
    body = text(path, stream)
    if "--tail" in argv:
        i = argv.index("--tail")
        n = int(argv[i + 1]) if i + 1 < len(argv) else 40
        body = "\n".join(body.splitlines()[-n:])
    print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
