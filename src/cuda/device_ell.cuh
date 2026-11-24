#pragma once

#include "device_buffer.cuh"
#include "device_views.cuh"

#include <sdnn/ell.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace sdnn::cuda {

struct DeviceEllLayers {
  std::int32_t neurons = 0;
  std::int32_t width = 0;
  std::int32_t num_layers = 0;
  DeviceBuffer<std::int32_t> indices;
  DeviceBuffer<float> values;

  explicit DeviceEllLayers(const EllLayers &host)
      : neurons{host.neurons}, width{host.width}, num_layers{host.num_layers},
        indices{host.indices.size()}, values{host.values.size()} {
    indices.upload(host.indices);
    values.upload(host.values);
  }

  auto layer_size() const -> std::size_t {
    return static_cast<std::size_t>(neurons) * static_cast<std::size_t>(width);
  }

  auto layer(std::size_t layer_idx) const -> EllView {
    if (layer_idx >= static_cast<std::size_t>(num_layers)) {
      throw std::out_of_range("layer index out of range");
    }
    const auto offset = layer_size() * layer_idx;
    return {
        .rows = neurons,
        .width = width,
        .indices = indices.span().subspan(offset, layer_size()),
        .values = values.span().subspan(offset, layer_size()),
    };
  }
};

} // namespace sdnn::cuda
