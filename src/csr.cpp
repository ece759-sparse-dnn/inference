#include <sdnn/csr.hpp>

namespace sdnn {

auto CsrLayers::layer_indptr(std::size_t layer_idx) const
    -> std::span<const std::int32_t> {
  auto stride = static_cast<std::size_t>(neurons) + 1;
  return std::span(indptr).subspan(stride * layer_idx, stride);
}

auto CsrLayers::layer_indices(std::size_t layer_idx) const
    -> std::span<const std::int32_t> {
  auto stride = static_cast<std::size_t>(nnz_per_layer);
  return std::span(indices).subspan(stride * layer_idx, stride);
}

auto CsrLayers::layer_values(std::size_t layer_idx) const
    -> std::span<const float> {
  auto stride = static_cast<std::size_t>(nnz_per_layer);
  return std::span(values).subspan(stride * layer_idx, stride);
}

auto CsrLayers::layer(std::size_t layer_idx) const -> CsrView {
  return {
      .rows = neurons,
      .cols = neurons,
      .indptr = layer_indptr(layer_idx),
      .indices = layer_indices(layer_idx),
      .values = layer_values(layer_idx),
  };
}

} // namespace sdnn
