/**
 * The generic interface that backends implement.
 */

#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "sdnn/dataset.hpp"

namespace sdnn {

using InferFn = std::vector<std::int32_t> (*)(const Dataset &, int);

struct Backend {
  std::string_view name;
  InferFn infer;
};

std::vector<std::int32_t> infer_serial(const Dataset &dataset, int layers);
std::vector<std::int32_t> infer_openmp(const Dataset &dataset, int layers);
#ifdef CUDA_ENABLED
std::vector<std::int32_t> infer_cuda(const Dataset &dataset, int layers);
#endif

std::span<const Backend> backends();

} // namespace sdnn
