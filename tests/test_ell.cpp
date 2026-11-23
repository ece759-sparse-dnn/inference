#include <gtest/gtest.h>

#include <sdnn/csr.hpp>
#include <sdnn/ell.hpp>

#include <stdexcept>
#include <vector>

namespace {

sdnn::Csr tiny_csr() {
  return {
      .rows = 3,
      .cols = 3,
      .indptr = {0, 2, 4, 6},
      .indices = {0, 1, 1, 2, 0, 2},
      .values = {1, 2, 3, 4, 5, 6},
  };
}

sdnn::CsrView view_of(const sdnn::Csr &csr) {
  return {
      .rows = csr.rows,
      .cols = csr.cols,
      .indptr = csr.indptr,
      .indices = csr.indices,
      .values = csr.values,
  };
}

} // namespace

TEST(Ell, FromTranspose) {
  const auto csr = tiny_csr();
  const auto ell = sdnn::ell::from_transpose(view_of(csr));

  EXPECT_EQ(ell.rows, 3);
  EXPECT_EQ(ell.width, 2);
  EXPECT_EQ(ell.indices, (std::vector<std::int32_t>{0, 0, 1, 2, 1, 2}));
  EXPECT_EQ(ell.values, (std::vector<float>{1, 2, 4, 5, 3, 6}));
}

TEST(Ell, EmplaceOverwritesOutput) {
  const auto csr = tiny_csr();
  std::vector<std::int32_t> idx(6, -1);
  std::vector<float> val(6, -1.0f);

  sdnn::ell::emplace_from_transpose(view_of(csr), idx, val);

  EXPECT_EQ(idx, (std::vector<std::int32_t>{0, 0, 1, 2, 1, 2}));
  EXPECT_EQ(val, (std::vector<float>{1, 2, 4, 5, 3, 6}));
}

TEST(Ell, IrregularColumnsThrow) {
  sdnn::Csr csr = tiny_csr();
  csr.indices = {0, 1, 0, 2, 0, 2};
  EXPECT_THROW(sdnn::ell::from_transpose(view_of(csr)), std::invalid_argument);
}

TEST(Ell, OutOfRangeColumnThrows) {
  sdnn::Csr csr = tiny_csr();
  csr.indices[0] = -1;
  EXPECT_THROW(sdnn::ell::from_transpose(view_of(csr)), std::invalid_argument);
}

TEST(Ell, WrongOutputSizeThrows) {
  const auto csr = tiny_csr();
  std::vector<std::int32_t> idx(5);
  std::vector<float> val(5);
  EXPECT_THROW(sdnn::ell::emplace_from_transpose(view_of(csr), idx, val),
               std::invalid_argument);
}

TEST(Ell, LayersFromTranspose) {
  const auto csr = tiny_csr();
  sdnn::CsrLayers layers{
      .neurons = 3,
      .num_layers = 2,
      .nnz_per_layer = 6,
  };
  for (int l = 0; l < 2; ++l) {
    layers.indptr.insert(layers.indptr.end(), csr.indptr.begin(),
                         csr.indptr.end());
    layers.indices.insert(layers.indices.end(), csr.indices.begin(),
                          csr.indices.end());
    for (float v : csr.values) {
      layers.values.push_back(v + 10.0f * l);
    }
  }

  const auto ell = sdnn::ell::from_transpose(layers, 2);

  EXPECT_EQ(ell.neurons, 3);
  EXPECT_EQ(ell.width, 2);
  EXPECT_EQ(ell.num_layers, 2);
  EXPECT_EQ(ell.indices,
            (std::vector<std::int32_t>{0, 0, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2}));
  EXPECT_EQ(ell.values,
            (std::vector<float>{1, 2, 4, 5, 3, 6, 11, 12, 14, 15, 13, 16}));

  const auto second = ell.layer(1);
  EXPECT_EQ(second.values[0], 11.0f);
}

TEST(Ell, TooManyLayersThrows) {
  sdnn::CsrLayers layers{.neurons = 3, .num_layers = 1, .nnz_per_layer = 6};
  EXPECT_THROW(sdnn::ell::from_transpose(layers, 2), std::invalid_argument);
}
