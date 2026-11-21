#include <stdexcept>

#include "sdnn/backend.hpp"

namespace sdnn {

std::vector<std::int32_t> infer_cuda(const Dataset &, int) {
  throw sdnn::not_implemented("cuda backend not implemented");
}

} // namespace sdnn
