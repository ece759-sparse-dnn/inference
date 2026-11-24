#pragma once

#include <cuda/std/span>

#include <cstdint>

namespace sdnn::cuda {

struct EllView {
  std::int32_t rows = 0;
  std::int32_t width = 0;
  ::cuda::std::span<const std::int32_t> indices;
  ::cuda::std::span<const float> values;
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

} // namespace sdnn::cuda
