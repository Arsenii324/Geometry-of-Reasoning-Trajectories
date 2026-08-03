#!/bin/bash
# Wait for a free Kaggle slot (max 2 concurrent), pull finished outputs, launch rho-direct.
# Bounded: polls at most MAX_POLLS times, launches exactly ONE kernel, then exits.
cd "$(dirname "$0")/../.." || exit 1
RUNNING=(geometry-causal-patch geometry-untrained-baseline)
MAX_POLLS=120        # 120 * 60s = 2h ceiling, then give up rather than poll forever
LAUNCHED=0

for ((i=0; i<MAX_POLLS; i++)); do
  busy=0
  for k in "${RUNNING[@]}"; do
    st=$(kaggle kernels status "arsen4ikvar/$k" 2>&1 | grep -o 'Status\.[A-Z]*')
    if [ "$st" = "Status.RUNNING" ] || [ "$st" = "Status.QUEUED" ]; then
      busy=$((busy+1))
    else
      d="scratch/kaggle_${k#geometry-}"; d="${d//-/_}"
      if [ -d "$d" ] && [ ! -f "$d/.pulled" ]; then
        echo "[$(date +%H:%M:%S)] $k -> $st, pulling output into $d/out"
        mkdir -p "$d/out" && kaggle kernels output "arsen4ikvar/$k" -p "$d/out" >/dev/null 2>&1 \
          && touch "$d/.pulled" && echo "[$(date +%H:%M:%S)] pulled $k"
      fi
    fi
  done

  if [ "$busy" -lt 2 ] && [ "$LAUNCHED" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] slot free ($busy busy) -- launching geometry-rho-direct"
    if kaggle kernels push -p scratch/kaggle_rho_direct; then
      LAUNCHED=1
      echo "[$(date +%H:%M:%S)] launched"
    else
      echo "[$(date +%H:%M:%S)] push FAILED -- not retrying, will exit"
      exit 1
    fi
  fi

  if [ "$busy" -eq 0 ] && [ "$LAUNCHED" -eq 1 ]; then
    echo "[$(date +%H:%M:%S)] all prior kernels finished and rho-direct is launched; done watching"
    exit 0
  fi
  sleep 60
done
echo "[$(date +%H:%M:%S)] poll ceiling reached; exiting (launched=$LAUNCHED)"
