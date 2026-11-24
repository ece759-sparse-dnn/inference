#include "kernels.cuh"

namespace sdnn::cuda {

namespace {

__device__ auto relu(float x, float mx) -> float {
  return fmaxf(0.0f, fminf(x, mx));
}

} // namespace

__global__ auto ell_relu_kernel(float ymax, float bias, EllView layer,
                                ActivationsView yin, ActivationsRef yout)
    -> void {
  const auto img = static_cast<std::int64_t>(blockIdx.y);
  const auto row =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (img >= yin.rows || row >= layer.rows) {
    return;
  }

  const auto in = img * yin.cols;
  float sum = 0.0f;
  for (std::int32_t t = 0; t < layer.width; ++t) {
    const auto i = static_cast<std::int64_t>(t) * layer.rows + row;
    sum += layer.values[i] * yin.values[in + layer.indices[i]];
  }

  yout.values[img * yout.cols + row] =
      sum > 0.0f ? relu(sum + bias, ymax) : 0.0f;
}

__global__ auto active_images_kernel(ActivationsView y,
                                     ::cuda::std::span<std::int32_t> active)
    -> void {
  const auto img = static_cast<std::int64_t>(blockIdx.x);
  if (img >= y.rows) {
    return;
  }

  const auto base = img * y.cols;
  int any = 0;
  for (auto col = static_cast<std::int64_t>(threadIdx.x); col < y.cols;
       col += blockDim.x) {
    any |= y.values[base + col] != 0.0f;
  }

  any = __syncthreads_or(any);
  if (threadIdx.x == 0) {
    active[img] = any != 0;
  }
}

} // namespace sdnn::cuda
