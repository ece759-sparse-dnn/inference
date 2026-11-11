#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <vector>

namespace sdnn {

inline constexpr int kBinVersion = 1;

struct Csr {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::vector<std::int32_t> indptr;
  std::vector<std::int32_t> indices;
  std::vector<float> values;
};

struct Dataset {
  std::int32_t neurons = 0;
  std::int32_t num_layers = 0;
  std::int32_t nnz_per_layer = 0;
  float bias = 0.0f;
  float ymax = 0.0f;
  Csr images;
  std::vector<std::int32_t> weights_indptr;
  std::vector<std::int32_t> weights_indices;
  std::vector<float> weights_values;
  std::map<int, std::vector<std::int32_t>> categories;

  std::span<const std::int32_t> layer_indptr(int i) const {
    const std::size_t n = static_cast<std::size_t>(neurons) + 1;
    return std::span(weights_indptr).subspan(i * n, n);
  }

  std::span<const std::int32_t> layer_indices(int i) const {
    const std::size_t n = static_cast<std::size_t>(nnz_per_layer);
    return std::span(weights_indices).subspan(i * n, n);
  }

  std::span<const float> layer_values(int i) const {
    const std::size_t n = static_cast<std::size_t>(nnz_per_layer);
    return std::span(weights_values).subspan(i * n, n);
  }
};

Dataset load_dataset(const std::filesystem::path &dir);

} // namespace sdnn
