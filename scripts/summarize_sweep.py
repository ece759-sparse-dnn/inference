#!/usr/bin/env python3
"""Summarize a schedule_sweep.sh CSV.

Prints median seconds per (schedule, threads), speedup over the serial
backend (rows whose backend is "serial"), and the best schedule per thread
count. Uses only the standard library so it runs on a bare cluster python3.

Usage: summarize_sweep.py SWEEP_CSV
"""

import csv
import statistics
import sys
from collections import defaultdict


def main(path: str) -> int:
    times = defaultdict(list)  # (schedule, threads) -> [seconds]
    serial = []
    failed = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["pass"] != "true":
                failed += 1
            seconds = float(row["seconds"])
            if row["backend"] == "serial":
                serial.append(seconds)
            else:
                times[(row["schedule"], int(row["threads"]))].append(seconds)

    if not times:
        print("no openmp rows found")
        return 1

    schedules = sorted({s for s, _ in times}, key=lambda s: (s.split(",")[0], s))
    threads = sorted({t for _, t in times})
    median = {key: statistics.median(v) for key, v in times.items()}
    base = statistics.median(serial) if serial else None

    width = max(len(s) for s in schedules) + 2

    def table(title, cell):
        print(f"\n{title}")
        print("schedule".ljust(width) + "".join(f"{t:>9}" for t in threads))
        for s in schedules:
            cells = []
            for t in threads:
                value = cell(s, t)
                cells.append(f"{value:>9}" if value is not None else f"{'-':>9}")
            print(s.ljust(width) + "".join(cells))

    reps = max(len(v) for v in times.values())
    table(
        f"median seconds (up to {reps} reps)",
        lambda s, t: f"{median[(s, t)]:.3f}" if (s, t) in median else None,
    )

    if 1 in threads:
        table(
            "scaling: speedup over the same schedule at 1 thread",
            lambda s, t: f"{median[(s, 1)] / median[(s, t)]:.2f}x"
            if (s, 1) in median and (s, t) in median
            else None,
        )

    if base is not None:
        print(f"\nserial baseline: {base:.3f} s (median of {len(serial)})")
        print("(only meaningful if serial and openmp run the same algorithm)")
        table(
            "speedup over serial",
            lambda s, t: f"{base / median[(s, t)]:.2f}x" if (s, t) in median else None,
        )
        table(
            "parallel efficiency (speedup / threads)",
            lambda s, t: f"{base / median[(s, t)] / t:.2f}" if (s, t) in median else None,
        )

    print("\nbest schedule per thread count")
    for t in threads:
        candidates = [(median[(s, t)], s) for s in schedules if (s, t) in median]
        best_time, best = min(candidates)
        extra = f"  ({base / best_time:.2f}x)" if base is not None else ""
        print(f"  {t:>3} threads: {best:<{width}} {best_time:.3f} s{extra}")

    if failed:
        print(f"\nWARNING: {failed} row(s) did not pass")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
