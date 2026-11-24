#pragma once

#include <sdnn/csr.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sdnn {

struct Ell {
  std::int32_t rows = 0;
  std::int32_t width = 0;
  std::vector<std::int32_t> indices;
  std::vector<float> values;
};

struct EllView {
  std::int32_t rows = 0;
  std::int32_t width = 0;
  std::span<const std::int32_t> indices;
  std::span<const float> values;
};

struct EllLayers {
  std::int32_t neurons = 0;
  std::int32_t width = 0;
  std::int32_t num_layers = 0;
  std::vector<std::int32_t> indices;
  std::vector<float> values;

  auto layer_size() const -> std::size_t;

  auto layer_indices(std::size_t layer_idx) const
      -> std::span<const std::int32_t>;

  auto layer_indices(std::size_t layer_idx) -> std::span<std::int32_t>;

  auto layer_values(std::size_t layer_idx) const -> std::span<const float>;

  auto layer_values(std::size_t layer_idx) -> std::span<float>;

  auto layer(std::size_t layer_idx) const -> EllView;
};

namespace ell {

auto width_of(CsrView layer) -> std::int32_t;

auto emplace_from_transpose(CsrView layer, std::span<std::int32_t> idx,
                            std::span<float> val) -> void;

auto from_transpose(CsrView layer) -> Ell;

auto from_transpose(const CsrLayers &layers, std::size_t num_layers)
    -> EllLayers;

} // namespace ell

} // namespace sdnn
