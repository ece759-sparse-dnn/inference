/**
 * The generic interface that backends implement.
 */

#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "sdnn/dataset.hpp"

namespace sdnn {

struct not_implemented : std::logic_error {
  using std::logic_error::logic_error;
};

struct CudaOptions {
  // Zero disables compaction. A positive value packs active rows after that
  // layer.
  int compact_after = 0;
};

struct InferOptions {
  CudaOptions cuda;
};

using InferFn = std::vector<std::int32_t> (*)(const Dataset &, int,
                                              const InferOptions &);

struct Backend {
  std::string_view name;
  InferFn infer;
  InferOptions preset{};
};

std::vector<std::int32_t> infer_serial(const Dataset &dataset, int layers,
                                       const InferOptions &options);
std::vector<std::int32_t> infer_openmp(const Dataset &dataset, int layers,
                                       const InferOptions &options);
#ifdef CUDA_ENABLED
std::vector<std::int32_t> infer_cuda(const Dataset &dataset, int layers,
                                     const InferOptions &options);
#endif

std::span<const Backend> backends();

// Sets the option named by a dotted key, e.g. "cuda.compact_after". Throws
// std::invalid_argument for an unknown key or a malformed value.
void set_option(InferOptions &options, std::string_view key,
                std::string_view value);

} // namespace sdnn
