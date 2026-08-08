#!/usr/bin/env bash
# DataSphere entrypoint:  bash ${RUN} ${JOB} ${RESULT}
#
# Stateless by design, following the pattern in the reference archive: install,
# run, tar everything back. `outputs:` only uploads when the job ENDS, so the job
# is kept short and writes its JSON incrementally -- a job killed by a platform
# limit otherwise loses the lot.
set -uo pipefail
JOB=${1:?job script}; RESULT=${2:?result tarball}
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
nproc; free -g | awk 'NR==2{print "RAM "$2"G total, "$7"G avail"}'

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq python3 python3-pip >/dev/null 2>&1
python3 -m pip -q install --upgrade pip >/dev/null 2>&1
# cu121 matches the 12.2.2 runtime image. transformers is pinned to the window the
# custom Huginn modeling code needs: outside 4.50-4.53 it breaks (constants.py).
python3 -m pip -q install torch --index-url https://download.pytorch.org/whl/cu121 >/dev/null 2>&1
python3 -m pip -q install "transformers>=4.50,<4.54" scipy numpy wandb >/dev/null 2>&1
python3 -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())'

OUTDIR=$(mktemp -d)
export RESULT_DIR="$OUTDIR"
python3 "$JOB" 2>&1 | tee "$OUTDIR/stdout.log"
rc=${PIPESTATUS[0]}
echo "job exit=$rc after $(( $(date +%s) - t0 ))s"
tar -czf "$RESULT" -C "$OUTDIR" .
exit 0   # always upload the tarball, even on failure -- the log is the evidence
