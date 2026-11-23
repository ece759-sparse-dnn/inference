#include "sdnn/dense.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace sdnn {

auto Dense::size() const -> std::size_t {
  return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

namespace dense {

auto emplace_from_csr(CsrView csr, std::span<float> out) -> void {
  const auto rows = static_cast<std::size_t>(csr.rows);
  const auto cols = static_cast<std::size_t>(csr.cols);
  if (csr.indptr.size() != rows + 1 || out.size() != rows * cols) {
    throw std::invalid_argument("malformed input: inconsistent dimensions");
  }

  std::fill(out.begin(), out.end(), 0.0f);

  for (std::size_t row_idx{}; row_idx < rows; ++row_idx) {
    auto row_start = csr.indptr[row_idx], row_end = csr.indptr[row_idx + 1];
    if (row_start < 0 || row_end < row_start ||
        static_cast<std::size_t>(row_end) > csr.indices.size()) {
      throw std::invalid_argument("malformed input: bad indptr");
    }
    for (auto csr_idx{row_start}; csr_idx < row_end; ++csr_idx) {
      auto col_idx = csr.indices[csr_idx];
      if (col_idx < 0 || col_idx >= csr.cols) {
        throw std::invalid_argument("malformed input");
      }
      out[row_idx * cols + static_cast<std::size_t>(col_idx)] =
          csr.values[csr_idx];
    }
  }
}

auto from_csr(CsrView csr) -> Dense {
  Dense result{.rows = csr.rows, .cols = csr.cols};
  result.values.resize(result.size());

  emplace_from_csr(csr, result.values);

  return result;
}

auto from_csr(const Csr &csr) -> Dense {
  return from_csr(CsrView{
      .rows = csr.rows,
      .cols = csr.cols,
      .indptr = csr.indptr,
      .indices = csr.indices,
      .values = csr.values,
  });
}

} // namespace dense

} // namespace sdnn
