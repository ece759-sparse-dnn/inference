# inference

https://graphchallenge.mit.edu/challenges/

## Dependencies

- GCC (15.2)
- CMake (Minimum 3.24)
- CUDA (13.0.0)
- Python (3.12)
- UV (0.13)

## Building

### Debug
```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

## Downloading Datasets
> [!WARNING]
> The data sets for 4096 Neurons onwards are quite large.

Info:
```bash
uv run --project python sdnn-tools -h
```

## Slurm Module Setup
```bash
module purge
module load gnu15/15.2.0 nvidia/cuda/13.0.0
```
