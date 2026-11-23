#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sdnn {

/**
 * A CSR matrix
 */
struct Csr {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::vector<std::int32_t> indptr;
  std::vector<std::int32_t> indices;
  std::vector<float> values;
};

/**
 * Non-owning view of a CSR matrix
 */
struct CsrView {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::span<const std::int32_t> indptr;
  std::span<const std::int32_t> indices;
  std::span<const float> values;
};

/**
 * Data for a series of CSR matrices corresponding to NN layers
 */
struct CsrLayers {
  std::int32_t neurons = 0;
  std::int32_t num_layers = 0;
  std::int32_t nnz_per_layer = 0;
  std::vector<std::int32_t> indptr;
  std::vector<std::int32_t> indices;
  std::vector<float> values;

  auto layer_indptr(std::size_t layer_idx) const
      -> std::span<const std::int32_t>;

  auto layer_indices(std::size_t layer_idx) const
      -> std::span<const std::int32_t>;

  auto layer_values(std::size_t layer_idx) const -> std::span<const float>;

  auto layer(std::size_t layer_idx) const -> CsrView;
};

} // namespace sdnn
