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

} // namespace

std::vector<std::int32_t> infer_cuda(const Dataset &d, int layers) {
  const auto num_layers = static_cast<std::size_t>(layers);
  const auto neurons = d.layers.neurons;
  const auto rows = d.images.rows;

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
  const dim3 grid((static_cast<unsigned>(neurons) + kBlockSize - 1) /
                      kBlockSize,
                  static_cast<unsigned>(rows));

  for (std::size_t l{}; l < num_layers; ++l) {
    cuda::ell_relu_kernel<<<grid, block>>>(
        d.ymax, d.bias, weights.layer(l),
        {.rows = rows, .cols = neurons, .values = yin.span()},
        {.rows = rows, .cols = neurons, .values = yout.span()});
    cuda::check(cudaGetLastError());
    std::swap(yin, yout);
  }

  cuda::DeviceBuffer<std::int32_t> d_active(static_cast<std::size_t>(rows));
  cuda::active_images_kernel<<<static_cast<unsigned>(rows), block>>>(
      {.rows = rows, .cols = neurons, .values = yin.span()}, d_active.span());
  cuda::check(cudaGetLastError());

  std::vector<std::int32_t> active(static_cast<std::size_t>(rows));
  d_active.download(active);

  std::vector<std::int32_t> result;
  for (std::int32_t img{}; img < rows; ++img) {
    if (active[img] != 0) {
      result.push_back(img);
    }
  }
  return result;
}

} // namespace sdnn
