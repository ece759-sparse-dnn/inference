#include <stdexcept>

#include "sdnn/backend.hpp"

namespace sdnn {

std::vector<std::int32_t> infer_openmp(const Dataset &, int) {
  throw std::logic_error("openmp backend not implemented");
}

} // namespace sdnn
