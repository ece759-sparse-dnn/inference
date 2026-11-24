#pragma once

#include <sdnn/csr.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sdnn {

/**
 * A dense row-major matrix
 */
struct Dense {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::vector<float> values;

  auto size() const -> std::size_t;
};

namespace dense {

auto emplace_from_csr(CsrView csr, std::span<float> out) -> void;

auto from_csr(CsrView csr) -> Dense;

auto from_csr(const Csr &csr) -> Dense;

} // namespace dense

} // namespace sdnn
