#include <stdexcept>

#include "sdnn/backend.hpp"

namespace sdnn {

std::vector<std::int32_t> infer_serial(const Dataset &, int) {
  throw std::logic_error("serial backend not implemented");
}

} // namespace sdnn
