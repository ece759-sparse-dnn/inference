"""Compare every single compaction point on the official 1024 x 120 case."""

import argparse
import csv
import os
import random
import statistics
import subprocess
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", help="path to the CUDA build of sdnn_run")
    parser.add_argument("output", help="new CSV file for all measurements")
    args = parser.parse_args()
    command = [args.binary, "--backend", "cuda", "--size", "1024", "--layers", "120"]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    rng = random.Random(759)
    timings = defaultdict(list)

    with open(args.output, "x", newline="") as output:
        writer = None

        def measure(phase, repeat, order):
            nonlocal writer
            for position, layer in enumerate(order, 1):
                flags = ["--set", f"cuda.compact_after={layer}"] if layer else []
                stdout = subprocess.check_output(command + flags, env=env, text=True)
                (row,) = csv.DictReader(stdout.splitlines())
                backend = f"cuda+cuda.compact_after={layer}" if layer else "cuda"
                observed = tuple(row[key] for key in
                                 ("backend", "neurons", "layers", "images", "threads", "pass"))
                if observed != (backend, "1024", "120", "60000", "1", "true"):
                    raise ValueError(f"Unexpected benchmark result: {row}")
                record = dict(phase=phase, round=repeat, position=position,
                              after_layer=layer, **row)
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=record)
                    writer.writeheader()
                writer.writerow(record)
                output.flush()
                timings[phase, layer].append(float(row["seconds"]))

        measure("warmup", 0, [0, 8])
        for repeat in range(1, 4):
            order = list(range(120))  # Zero is the uncompressed baseline.
            rng.shuffle(order)
            measure("screen", repeat, order)

        ranked = sorted(range(1, 120), key=lambda k: statistics.median(timings["screen", k]))
        finalists = sorted({0, 8, 10, *ranked[:5]})
        for cycle in range(2):
            rng.shuffle(finalists)
            for shift in range(len(finalists)):
                repeat = 1 + cycle * len(finalists) + shift
                measure("confirm", repeat, finalists[shift:] + finalists[:shift])

    for layer in sorted(finalists, key=lambda k: statistics.median(timings["confirm", k])):
        print(f"layer={layer:3d}  median={statistics.median(timings['confirm', layer]):.6f} s")


if __name__ == "__main__":
    main()
