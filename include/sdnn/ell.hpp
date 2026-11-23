#pragma once

#include "sdnn/dataset.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sdnn {

struct Ell {
  std::int32_t nnz_per_col = 0;
  std::vector<std::int32_t> indices;
  std::vector<float> values;
};

struct EllLayers {
  std::int32_t neurons = 0;
  std::int32_t num_layers = 0;
  std::int32_t nnz_per_col = 0;
  std::vector<std::int32_t> indices;
  std::vector<float> values;
};

auto csr_to_ell(CsrView layer, std::size_t nnz_per_col,
                std::span<std::int32_t> idx, std::span<float> val) -> void;

} // namespace sdnn
