#include <gtest/gtest.h>

#include <sdnn/dataset.hpp>
#include <sdnn/ell.hpp>

TEST(Ell, CsrToEll) {
  sdnn::Csr csr{
      .rows = 3,
      .cols = 3,
      .indptr = {0, 2, 4, 6},
      .indices = {0, 1, 1, 2, 0, 2},
      .values = {1, 2, 3, 4, 5, 6},
  };

  sdnn::CsrView csr_view{
      .rows = 3,
      .cols = 3,
      .indptr = csr.indptr,
      .indices = csr.indices,
      .values = csr.values,
  };

  sdnn::Ell expected_ell{
      .nnz_per_col = 2,
      .indices = {0, 0, 1, 2, 1, 2},
      .values = {1, 2, 4, 5, 3, 6},
  };

  sdnn::Ell got_ell{
      .nnz_per_col = 2,
      .indices = std::vector<std::int32_t>(6, 0),
      .values = std::vector<float>(6, 0),
  };

  sdnn::csr_to_ell(csr_view, 2, got_ell.indices, got_ell.values);

  EXPECT_EQ(expected_ell.values, got_ell.values);
  EXPECT_EQ(expected_ell.indices, got_ell.indices);
}
