#include "cuda/check.cuh"
#include "cuda/device_buffer.cuh"
#include "cuda/device_ell.cuh"
#include "cuda/device_views.cuh"
#include "cuda/kernels.cuh"
#include "sdnn/backend.hpp"
#include "sdnn/dense.hpp"
#include "sdnn/ell.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sdnn {

namespace {

constexpr unsigned kBlockSize = 256;
constexpr std::int32_t kMaxGridY = 65535;

std::vector<std::int32_t> active_rows(cuda::ActivationsView y) {
  cuda::DeviceBuffer<std::int32_t> flags(y.rows);
  cuda::active_images_kernel<<<y.rows, kBlockSize>>>(y, flags.span());
  cuda::check(cudaGetLastError());
  std::vector<std::int32_t> active(y.rows);
  flags.download(active);
  std::vector<std::int32_t> ids;
  for (std::int32_t i = 0; i < y.rows; ++i) {
    if (active[i]) {
      ids.push_back(i);
    }
  }
  return ids;
}

// Pack into the unused output buffer so unread source rows cannot be
// overwritten.
__global__ void gather_rows(cuda::ActivationsView src, const std::int32_t *ids,
                            float *dst) {
  const auto row = static_cast<std::int64_t>(blockIdx.x);
  const auto original = static_cast<std::int64_t>(ids[row]);
  for (auto col = static_cast<std::int64_t>(threadIdx.x); col < src.cols;
       col += blockDim.x) {
    dst[row * src.cols + col] = src.values[original * src.cols + col];
  }
}

} // namespace

std::vector<std::int32_t> infer_cuda_compact(const Dataset &d, int layers,
                                             int after_layer) {
  if (after_layer < 0 || (after_layer > 0 && after_layer >= layers)) {
    throw std::invalid_argument("compaction must precede the final layer");
  }
  const auto num_layers = static_cast<std::size_t>(layers);
  const auto neurons = d.layers.neurons;
  auto rows = d.images.rows;

  if (d.images.cols != neurons) {
    throw std::invalid_argument("image width must match neuron count");
  }
  if (rows > kMaxGridY) {
    throw std::invalid_argument("too many images for a single launch");
  }
  if (rows == 0 || neurons == 0) {
    return {};
  }

  const cuda::DeviceEllLayers weights(
      ell::from_transpose(d.layers, num_layers));

  const auto images = dense::from_csr(d.images);
  cuda::DeviceBuffer<float> yin(images.size());
  cuda::DeviceBuffer<float> yout(images.size());
  yin.upload(images.values);

  const dim3 block(kBlockSize);
  const auto blocks =
      (static_cast<unsigned>(neurons) + kBlockSize - 1) / kBlockSize;
  std::vector<std::int32_t> original_ids;

  for (std::size_t l{}; l < num_layers; ++l) {
    if (after_layer > 0 && l == static_cast<std::size_t>(after_layer)) {
      const cuda::ActivationsView y{rows, neurons, yin.span()};
      original_ids = active_rows(y);
      if (original_ids.empty()) {
        return {};
      }
      if (original_ids.size() < static_cast<std::size_t>(rows)) {
        rows = static_cast<std::int32_t>(original_ids.size());
        cuda::DeviceBuffer<std::int32_t> ids(rows);
        ids.upload(original_ids);
        gather_rows<<<rows, block>>>(y, ids.data(), yout.data());
        cuda::check(cudaGetLastError());
        std::swap(yin, yout);
      }
    }
    const dim3 grid(blocks, static_cast<unsigned>(rows));
    cuda::ell_relu_kernel<<<grid, block>>>(
        d.ymax, d.bias, weights.layer(l),
        {.rows = rows, .cols = neurons, .values = yin.span()},
        {.rows = rows, .cols = neurons, .values = yout.span()});
    cuda::check(cudaGetLastError());
    std::swap(yin, yout);
  }

  auto result = active_rows({rows, neurons, yin.span()});
  if (!original_ids.empty()) {
    for (auto &id : result) {
      id = original_ids[id];
    }
  }
  return result;
}

std::vector<std::int32_t> infer_cuda(const Dataset &d, int layers) {
  return infer_cuda_compact(d, layers, 0);
}

} // namespace sdnn
