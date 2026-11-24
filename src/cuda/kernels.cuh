/**
 * Kernel declarations, defined in kernels.cu and launched from cuda.cu.
 *
 * Launching a __global__ across translation units is fine without separable
 * compilation. __device__ helpers called from more than one .cu file must be
 * inline in a header (or the target needs CUDA_SEPARABLE_COMPILATION).
 */

#pragma once

#include "device_views.cuh"

namespace sdnn::cuda {

// TODO: ell_relu_kernel, active-image reduction

} // namespace sdnn::cuda
