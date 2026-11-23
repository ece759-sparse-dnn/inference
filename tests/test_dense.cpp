#include <gtest/gtest.h>

#include <sdnn/csr.hpp>
#include <sdnn/dense.hpp>

#include <span>
#include <stdexcept>
#include <vector>

namespace {

// [1 0 2]
// [0 0 0]
// [0 3 0]
sdnn::Csr tiny_csr() {
  return {
      .rows = 3,
      .cols = 3,
      .indptr = {0, 2, 2, 3},
      .indices = {0, 2, 1},
      .values = {1, 2, 3},
  };
}

} // namespace

TEST(Dense, FromCsr) {
  const auto dense = sdnn::dense::from_csr(tiny_csr());

  EXPECT_EQ(dense.rows, 3);
  EXPECT_EQ(dense.cols, 3);
  EXPECT_EQ(dense.size(), 9u);
  EXPECT_EQ(dense.values, (std::vector<float>{1, 0, 2, 0, 0, 0, 0, 3, 0}));
}

TEST(Dense, EmplaceOverwritesOutput) {
  const auto csr = tiny_csr();
  std::vector<float> out(9, -1.0f);

  sdnn::dense::emplace_from_csr({.rows = csr.rows,
                                 .cols = csr.cols,
                                 .indptr = csr.indptr,
                                 .indices = csr.indices,
                                 .values = csr.values},
                                out);

  EXPECT_EQ(out, (std::vector<float>{1, 0, 2, 0, 0, 0, 0, 3, 0}));
}

TEST(Dense, RowRange) {
  const auto csr = tiny_csr();
  // Rows [1, 3): indptr keeps absolute offsets into the full indices/values.
  const sdnn::CsrView rows_1_to_3{
      .rows = 2,
      .cols = csr.cols,
      .indptr = std::span(csr.indptr).subspan(1, 3),
      .indices = csr.indices,
      .values = csr.values,
  };

  const auto dense = sdnn::dense::from_csr(rows_1_to_3);

  EXPECT_EQ(dense.rows, 2);
  EXPECT_EQ(dense.values, (std::vector<float>{0, 0, 0, 0, 3, 0}));
}

TEST(Dense, OutOfRangeColumnThrows) {
  auto csr = tiny_csr();
  csr.indices[0] = 3;
  EXPECT_THROW(sdnn::dense::from_csr(csr), std::invalid_argument);
}

TEST(Dense, BadIndptrThrows) {
  auto csr = tiny_csr();
  csr.indptr = {0, 2, 2};
  EXPECT_THROW(sdnn::dense::from_csr(csr), std::invalid_argument);
}

TEST(Dense, WrongOutputSizeThrows) {
  const auto csr = tiny_csr();
  std::vector<float> out(8);
  EXPECT_THROW(sdnn::dense::emplace_from_csr({.rows = csr.rows,
                                              .cols = csr.cols,
                                              .indptr = csr.indptr,
                                              .indices = csr.indices,
                                              .values = csr.values},
                                             out),
               std::invalid_argument);
}
