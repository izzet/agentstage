# paper_figures/data/

Version-controlled data snapshots backing paper figures. Distinct from
`paper/figures/data/` (which is in `.gitignore` and only holds per-build
artefacts).

## Files

| File | Source script | What it backs |
|---|---|---|
| `bench_tiers_small_files_20260528.csv` | `scripts/bench_tiers.sh --small-files` | Agent-scale read-latency ladder (the AgentStage motivation table). Single-stream, 4 MiB files, 10 reps per tier. Cold cache via O_DIRECT (FUSE for orangefs/s3 by construction). |
| `bench_tiers_rigorous_20260528.csv` | `scripts/bench_tiers.sh --rigorous` | Steady-state cross-tier bandwidth ceilings + dd-vs-IOR cross-check. 4 MPI tasks, 4 GiB block, 5 reps per tier. |

Schema (both files):
```
tier,tool,op,odirect,tasks,n_reps,xfer,block,max_mibps,min_mibps,mean_mibps,stdev_mibps,mean_s,ms_per_4mib,test_file
```

See `scripts/BENCH_TIERS.md` for the headline reading and per-column docs.

## Regenerating

Both CSVs are reproducible from a single Slurm allocation:
```
salloc -N 1 -t 02:00:00 --exclusive
bash scripts/bench_tiers.sh --rigorous     # ~15-20 min
bash scripts/bench_tiers.sh --small-files  # ~3-5 min
```
Results land in `outputs/bench_tiers/<timestamp>/summary.csv` and can be
copied here when the run is the one a paper figure should rest on.

Naming convention: `bench_tiers_<mode>_<YYYYMMDD>.csv`. Bump the date
suffix when you re-snapshot — keep the historical CSVs so figure
provenance stays clear.
