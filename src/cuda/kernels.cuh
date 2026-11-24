#pragma once

#include "device_views.cuh"

#include <cuda/std/span>

#include <cstdint>

namespace sdnn::cuda {

__global__ auto ell_relu_kernel(float ymax, float bias, EllView layer,
                                ActivationsView yin, ActivationsRef yout)
    -> void;

__global__ auto active_images_kernel(ActivationsView y,
                                     ::cuda::std::span<std::int32_t> active)
    -> void;

__global__ auto gather_rows_kernel(ActivationsView src,
                                   ::cuda::std::span<const std::int32_t> ids,
                                   ActivationsRef dst) -> void;

} // namespace sdnn::cuda
