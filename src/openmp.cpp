#include <stdexcept>

#include "sdnn/backend.hpp"

namespace sdnn {

std::vector<std::int32_t> infer_openmp(const Dataset &, int) {
  throw sdnn::not_implemented("openmp backend not implemented");
}

} // namespace sdnn
