#include "sdnn/backend.hpp"

namespace sdnn {

std::span<const Backend> backends() {
  static constexpr Backend all[] = {
      {"serial", infer_serial},
      {"openmp", infer_openmp},
#ifdef CUDA_ENABLED
      {"cuda", infer_cuda},
#endif
  };
  return all;
}

} // namespace sdnn
