#pragma once

#include <cuda_runtime.h>

#include <stdexcept>

namespace sdnn::cuda {

inline auto check(cudaError_t error) -> void {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

} // namespace sdnn::cuda
