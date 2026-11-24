#include "sdnn/backend.hpp"

#include <cuda/std/span>

namespace sdnn {

namespace cuda {

constexpr float YMAX = 32.0f;

__device__ auto relu(float x, float mx) -> float {
  return fmaxf(0, fminf(mx, x));
}

struct EllLayersView {
  std::int32_t rows = 0;
  std::int32_t width = 0;
  std::int32_t num_layers = 0;
  ::cuda::std::span<const std::int32_t> indices;
  ::cuda::std::span<const float> values;
};

struct EllView {
  std::int32_t rows = 0;
  std::int32_t width = 0;
  ::cuda::std::span<const std::int32_t> indices;
  ::cuda::std::span<const float> values;
};

struct Dense {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::vector<float> values;
};

struct ActivationsView {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  ::cuda::std::span<const float> values;
};

struct ActivationsRef {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  ::cuda::std::span<float> values;
};

} // namespace cuda

// Computes Y' for a single layer
//   Y' = ReLU(AY + B)
// We assume B fixed
__global__ void ell_relu_kernel(float ymax, float bias, cuda::EllView layer,
                                cuda::ActivationsView yin,
                                cuda::ActivationsRef yout) {
  auto img = blockIdx.y;
  auto row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < layer.rows) {
    float sum = 0.0f;
    for (unsigned int t = 0; t < layer.width; ++t) {
      unsigned int i = t * layer.rows + row;
      unsigned int col = layer.indices[i];
      float val = layer.values[i];
      auto off = static_cast<std::int64_t>(img) * layer.rows;
      sum += val * yin.values[off + col];
    }
    yout.values[img * layer.rows + row] = (sum > 0) * relu(sum + bias, ymax);
  }
}

__host__ auto activations_size(const Dataset &d) -> std::size_t {
  return static_cast<std::size_t>(d.images.rows) *
         static_cast<std::size_t>(d.layers.rows);
}

auto dense_from(const Csr &csr) -> Dense {
  Dense result{.rows = csr.rows,
               .cols = csr.cols,
               .values = std::vector<float>(static_cast<std::size_t>(csr.rows) *
                                            csr.cols)};

  for (std::size_t row_idx{}; row_idx < csr.rows; ++row_idx) {
    for (std::size_t id{csr.indptr[row_idx]}, id < csr.indptr[row_idx + 1],
         ++id) {
      auto col_idx = csr.indices[id];
      auto val = csr.values[id];
      result.values[row_idx][col_idx] = val;
    }
  }

  return result;
}

std::vector<std::int32_t> infer_cuda(const Dataset &d, std::size_t num_layers) {
  auto layers = ell::from_transpose(d.layers, num_layers);

  void *d_ptr;
  cudaMalloc(&d_ptr, activations_size(d));

  Dense dense = dense_From(d.images);

  cudaMemcpy(d_ptr, dense.values.data(), activations_size(d),
             cudaMemcpyHostToDevice);

  ActivationsRef y{.rows = d.images.rows,
                   .cols = d.layers.rows,
                   .values = ::cuda::std::span(d_ptr, activations_size(d))};

  for (std::size_t l{}; l < num_layers; ++l) {
    auto layer = layers.layer(l);
    ell_relu_kernel(d.ymax, d.bias, layer, ...);
  }

  throw sdnn::not_implemented("cuda backend not implemented");
}

} // namespace sdnn
