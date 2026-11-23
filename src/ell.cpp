#include "sdnn/ell.hpp"
#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace sdnn {

auto csr_to_ell(CsrView layer, std::size_t nnz_per_col,
                std::span<std::int32_t> idx, std::span<float> val) -> void {
  if (idx.size() != val.size() || val.size() != nnz_per_col * layer.cols) {
    throw std::invalid_argument("malformed input: inconsistent dimensions");
  }

  std::vector<std::size_t> filled(layer.cols);
  for (std::int32_t row_idx{}; row_idx < layer.rows; ++row_idx) {
    auto row_start = layer.indptr[row_idx], row_end = layer.indptr[row_idx + 1];
    for (auto csr_idx{row_start}; csr_idx < row_end; ++csr_idx) {
      auto col_idx = layer.indices[csr_idx];
      if (col_idx >= layer.cols || filled[col_idx] >= nnz_per_col) {
        throw std::invalid_argument("malformed input");
      }
      auto ell_idx = col_idx + layer.cols * filled[col_idx]++;
      idx[ell_idx] = row_idx;
      val[ell_idx] = layer.values[csr_idx];
    }
  }

  if (!std::all_of(begin(filled), end(filled),
                   [=](std::size_t cnt) { return cnt == nnz_per_col; })) {
    throw std::invalid_argument("malformed input: nnz_per_col is inconsistent");
  }
}

} // namespace sdnn
