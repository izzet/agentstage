"""Decompression-staging transcoder (E-029).

Reads zlib-compressed NetCDF files from a cold tier and writes
data-identical UNCOMPRESSED copies into the hot tier, mirroring the
absolute-path layout the AgentStage shim expects
(hot_root / cold_abs_path.lstrip('/')).

Run under SYSTEM python3 (needs netCDF4). Invoked as a subprocess by
scripts/microbench/path_b_e2e_decompress.py.

Usage:
    python3 transcode.py <cold_dir> <hot_root> [n_workers]
"""

import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import netCDF4 as nc


def transcode_one(args: tuple[str, str]) -> tuple[str, int, int, bool, str]:
    """Transcode one file: read (decompress), write uncompressed.
    Returns (dst, src_bytes, dst_bytes, ok, error)."""
    src, dst = args
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        with nc.Dataset(src, "r") as s, \
             nc.Dataset(tmp, "w", format=s.data_model) as d:
            # global attributes
            d.setncatts({k: s.getncattr(k) for k in s.ncattrs()})
            # dimensions
            for name, dim in s.dimensions.items():
                d.createDimension(
                    name, None if dim.isunlimited() else len(dim))
            # variables — written with zlib=False (no compression filter)
            for name, var in s.variables.items():
                chunking = var.chunking()
                chunksizes = chunking if isinstance(chunking, list) else None
                fill = None
                if "_FillValue" in var.ncattrs():
                    fill = var.getncattr("_FillValue")
                out = d.createVariable(
                    name, var.dtype, var.dimensions,
                    zlib=False, complevel=0,
                    chunksizes=chunksizes, fill_value=fill,
                )
                out.setncatts({k: var.getncattr(k) for k in var.ncattrs()
                               if k != "_FillValue"})
                out[:] = var[:]
        os.replace(tmp, dst)
        return (dst, os.path.getsize(src), os.path.getsize(dst), True, "")
    except Exception as e:  # noqa: BLE001
        return (dst, 0, 0, False, repr(e))


def main() -> int:
    cold_dir = sys.argv[1]
    hot_root = sys.argv[2].rstrip("/")
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    files = sorted(set(
        glob.glob(os.path.join(cold_dir, "**", "*C0[8-9]*.nc"), recursive=True)
        + glob.glob(os.path.join(cold_dir, "**", "*C10*.nc"), recursive=True)
    ))
    jobs = []
    for src in files:
        abs_src = os.path.abspath(src)
        dst = os.path.join(hot_root, abs_src.lstrip("/"))
        jobs.append((abs_src, dst))

    print(f"transcode: {len(jobs)} files, {n_workers} workers", flush=True)
    t0 = time.monotonic()
    ok = 0
    src_total = dst_total = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for dst, sb, db, success, err in ex.map(transcode_one, jobs):
            if success:
                ok += 1
                src_total += sb
                dst_total += db
            else:
                print(f"  ERROR {dst}: {err}", file=sys.stderr, flush=True)
    elapsed = time.monotonic() - t0
    print(f"transcode done: {ok}/{len(jobs)} ok in {elapsed:.1f}s", flush=True)
    print(f"  src_total={src_total/1024/1024:.0f}MB  "
          f"dst_total={dst_total/1024/1024:.0f}MB  "
          f"expansion={dst_total/max(1,src_total):.2f}x", flush=True)
    return 0 if ok == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
