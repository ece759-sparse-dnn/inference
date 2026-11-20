#include <stdexcept>

#include "sdnn/backend.hpp"

namespace sdnn {

std::vector<std::int32_t> infer_serial(const Dataset &, int) {
  throw sdnn::not_implemented("serial backend not implemented");
}

} // namespace sdnn
