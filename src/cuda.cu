#include <stdexcept>

#include "sdnn/backend.hpp"

namespace sdnn {

std::vector<std::int32_t> infer_cuda(const Dataset &, int) {
  throw std::logic_error("cuda backend not implemented");
}

} // namespace sdnn
