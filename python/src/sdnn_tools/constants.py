"""Project-wide constants."""

SIZES: list[int] = [1024, 4096, 16384, 65536]
LAYERS: list[int] = [120, 480, 1920]
BIAS: dict[int, float] = {1024: -0.3, 4096: -0.35, 16384: -0.4, 65536: -0.45}
YMAX: float = 32.0
NNZ_PER_ROW: int = 32
BIN_VERSION: int = 1
