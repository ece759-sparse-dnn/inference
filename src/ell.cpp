#include "sdnn/ell.hpp"
#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace sdnn {

auto EllLayers::layer_size() const -> std::size_t {
  return static_cast<std::size_t>(neurons) * static_cast<std::size_t>(width);
}

auto EllLayers::layer_indices(std::size_t layer_idx) const
    -> std::span<const std::int32_t> {
  return std::span(indices).subspan(layer_size() * layer_idx, layer_size());
}

auto EllLayers::layer_indices(std::size_t layer_idx)
    -> std::span<std::int32_t> {
  return std::span(indices).subspan(layer_size() * layer_idx, layer_size());
}

auto EllLayers::layer_values(std::size_t layer_idx) const
    -> std::span<const float> {
  return std::span(values).subspan(layer_size() * layer_idx, layer_size());
}

auto EllLayers::layer_values(std::size_t layer_idx) -> std::span<float> {
  return std::span(values).subspan(layer_size() * layer_idx, layer_size());
}

auto EllLayers::layer(std::size_t layer_idx) const -> EllView {
  return {
      .rows = neurons,
      .width = width,
      .indices = layer_indices(layer_idx),
      .values = layer_values(layer_idx),
  };
}

namespace ell {

auto width_of(CsrView layer) -> std::int32_t {
  const auto nnz = layer.indices.size();
  const auto cols = static_cast<std::size_t>(layer.cols);
  if (cols == 0 || nnz % cols != 0) {
    throw std::invalid_argument(
        "malformed input: nnz is not a multiple of cols");
  }
  return static_cast<std::int32_t>(nnz / cols);
}

auto emplace_from_transpose(CsrView layer, std::span<std::int32_t> idx,
                            std::span<float> val) -> void {
  const auto width = static_cast<std::size_t>(width_of(layer));
  const auto cols = static_cast<std::size_t>(layer.cols);
  if (idx.size() != width * cols || val.size() != width * cols) {
    throw std::invalid_argument("malformed input: inconsistent dimensions");
  }

  std::vector<std::size_t> filled(cols);
  for (std::int32_t row_idx{}; row_idx < layer.rows; ++row_idx) {
    auto row_start = layer.indptr[row_idx], row_end = layer.indptr[row_idx + 1];
    for (auto csr_idx{row_start}; csr_idx < row_end; ++csr_idx) {
      auto col_idx = layer.indices[csr_idx];
      if (col_idx < 0 || col_idx >= layer.cols || filled[col_idx] >= width) {
        throw std::invalid_argument("malformed input");
      }
      auto ell_idx =
          static_cast<std::size_t>(col_idx) + cols * filled[col_idx]++;
      idx[ell_idx] = row_idx;
      val[ell_idx] = layer.values[csr_idx];
    }
  }

  if (!std::all_of(begin(filled), end(filled),
                   [=](std::size_t cnt) { return cnt == width; })) {
    throw std::invalid_argument("malformed input: nnz per col is inconsistent");
  }
}

auto from_transpose(CsrView layer) -> Ell {
  const auto width = width_of(layer);
  const auto size = static_cast<std::size_t>(layer.cols) * width;
  Ell result{
      .rows = layer.cols,
      .width = width,
      .indices = std::vector<std::int32_t>(size),
      .values = std::vector<float>(size),
  };

  emplace_from_transpose(layer, result.indices, result.values);

  return result;
}

auto from_transpose(const CsrLayers &layers, std::size_t num_layers)
    -> EllLayers {
  if (num_layers > static_cast<std::size_t>(layers.num_layers)) {
    throw std::invalid_argument("requested more layers than available");
  }
  if (layers.neurons == 0 || layers.nnz_per_layer % layers.neurons != 0) {
    throw std::invalid_argument(
        "malformed input: nnz_per_layer is not a multiple of neurons");
  }

  EllLayers result{
      .neurons = layers.neurons,
      .width = layers.nnz_per_layer / layers.neurons,
      .num_layers = static_cast<std::int32_t>(num_layers),
  };
  result.indices.resize(result.layer_size() * num_layers);
  result.values.resize(result.layer_size() * num_layers);

  for (std::size_t i{}; i < num_layers; ++i) {
    emplace_from_transpose(layers.layer(i), result.layer_indices(i),
                           result.layer_values(i));
  }

  return result;
}

} // namespace ell

} // namespace sdnn
