#!/usr/bin/env bash
# E-041 MLE-bench full-agentic-loop sweep — analog of dsbench_multiturn.py
# sweep used for E-040. 3 competitions × 2 modes (baseline/staged) × 3 reps
# = 18 sessions, Haiku 4.5, verified cold cache per session.
#
# Prerequisite: data must be prepared:
#   mlebench prepare -c <comp> --data-dir outputs/mlebench-data
#
# Usage:
#   bash scripts/microbench/mlebench_sweep.sh

set -uo pipefail

cd "$(dirname "$0")/../.."

TS=$(date +%Y%m%dT%H%M%S)
OUT="outputs/mlebench_mt/_sweep_$TS"
mkdir -p "$OUT"
echo "SWEEP=$TS"
echo "OUT=$OUT"
date

# 2 main + 1 honest-negative competitions (see workloads/mlebench.py
# for the rationale on each):
#   - nyc-taxi: 5.3 GB labels.csv, the natural CSV-heavy case
#   - dogs-vs-cats: 490 MB train.zip + 54 MB test.zip (zip access path)
#   - histopathologic: 220K-file unpacked dir; honest-negative case
COMPS=(
    new-york-city-taxi-fare-prediction
    dogs-vs-cats-redux-kernels-edition
    histopathologic-cancer-detection
)

for comp in "${COMPS[@]}"; do
    if [[ ! -d "outputs/mlebench-data/$comp/prepared/public" ]]; then
        echo "SKIPPING $comp — data not prepared (run mlebench prepare first)"
        continue
    fi
    for mode in baseline staged; do
        for rep in 1 2 3; do
            tag="${comp}_${mode}_r${rep}"
            outdir="$OUT/$tag"
            echo
            echo "===== $tag ($(date)) ====="
            ~/.local/bin/uv run python scripts/microbench/mlebench_multiturn.py \
                --task "$comp" \
                --model claude-haiku-4-5 \
                --mode "$mode" \
                --out "$outdir" \
                --max-turns 12 2>&1 | tail -5
        done
    done
done

echo
echo "=== SWEEP DONE $(date) ==="
echo "OUT=$OUT"

# Inline summary
~/.local/bin/uv run python -c "
import json, os, glob
from collections import defaultdict
sweep = '$OUT'
runs = defaultdict(list)
for d in sorted(glob.glob(f'{sweep}/*/')):
    sf = os.path.join(d, 'summary.json')
    if not os.path.isfile(sf): continue
    s = json.load(open(sf))
    runs[(s['task'], s['mode'])].append(s)
print()
print(f'{\"task\":<48} {\"mode\":<9} {\"sessions (s)\":<30} {\"median\":>8} {\"sub\":>5} {\"pf\":>4}')
print('-'*110)
agg = {}
for (task, mode), rs in sorted(runs.items()):
    sess = sorted(r['session_elapsed_s'] for r in rs)
    med = sess[len(sess)//2] if sess else 0
    subs = sum(1 for r in rs if r['submitted'])
    pfs = sum(r['n_prefetched_files'] for r in rs)
    agg[(task, mode)] = (sess, med, subs, len(rs), pfs)
    sess_str = ' / '.join(f'{s:.1f}' for s in sess)
    print(f'{task[:48]:<48} {mode:<9} {sess_str:<30} {med:>7.1f}s {subs}/{len(rs):<3} {pfs:>4}')
print()
print('=== Session speedup (baseline_median / staged_median) ===')
for t in sorted(set(t for (t,_) in runs.keys())):
    b = agg.get((t,'baseline')); s = agg.get((t,'staged'))
    if b and s and s[1] > 0:
        sp = b[1]/s[1]
        print(f'  {t:<48}  baseline={b[1]:>6.1f}s  staged={s[1]:>6.1f}s  speedup={sp:.2f}x')
" 2>&1 | tail -25
