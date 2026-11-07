SIZES: list[int] = [1024, 4096, 16384, 65536]
"""The number of neurons per layer a NN may have."""

DEPTHS: list[int] = [120, 480, 1920]
"""The number of layers a NN may have."""

BIAS: dict[int, float] = {1024: -0.3, 4096: -0.35, 16384: -0.4, 65536: -0.45}
"""The bias for a given NN size."""

YMAX: float = 32.0
"""The upper limit of the nonlinear transformation output."""

NNZ_PER_ROW: int = 32
"""The number of connections per neuron for RadiX-Net DNNs."""

BIN_VERSION: int = 1
"""The version of the serialization format."""
