#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "sdnn/backend.hpp"

class BackendTest : public testing::TestWithParam<sdnn::Backend> {
protected:
  static inline std::optional<sdnn::Dataset> dataset;

  static void SetUpTestSuite() {
    const char *env = std::getenv("SDNN_DATA_DIR");
    const std::filesystem::path root = env ? env : SDNN_DEFAULT_DATA_DIR;
    const auto dir = root / "bin" / "neuron1024";
    if (!std::filesystem::exists(dir / ".done")) {
      return;
    }
    dataset = sdnn::load_dataset(dir);
  }

  static void TearDownTestSuite() { dataset.reset(); }

  void SetUp() override {
    if (!dataset.has_value()) {
      GTEST_SKIP() << "no converted dataset";
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
    AllBackends, BackendTest, testing::ValuesIn(sdnn::backends()),
    [](const testing::TestParamInfo<BackendTest::ParamType> &info) {
      return std::string(info.param.name);
    });

TEST_P(BackendTest, MatchesTruth) {
  const auto &backend = GetParam();
  const auto &truth = dataset->categories.at(120);

  try {
    auto result = backend.infer(*dataset, 120, backend.preset);
    ASSERT_EQ(result.size(), truth.size());
    EXPECT_EQ(result, truth);
  } catch (const sdnn::not_implemented &e) {
    GTEST_SKIP() << e.what();
  }
}

#ifdef CUDA_ENABLED
TEST(CudaCompaction, PreservesOriginalIdsAsRowsDie) {
  sdnn::Dataset d;
  d.bias = -1;
  d.ymax = 32;
  d.images = {5, 1, {0, 0, 1, 2, 2, 3}, {0, 0, 0}, {11, 9, 13}};
  auto &w = d.layers;
  w.neurons = w.nnz_per_layer = 1;
  w.num_layers = 13;
  for (int l = 0; l < w.num_layers; ++l) {
    w.indptr.insert(w.indptr.end(), {0, 1});
    w.indices.push_back(0);
    w.values.push_back(1);
  }
  const auto compact = [&d](int layers, int after) {
    return sdnn::infer_cuda(d, layers, {.cuda = {.compact_after = after}});
  };
  const std::vector<std::int32_t> both{1, 4}, last{4};
  EXPECT_EQ(compact(10, 8), both);
  EXPECT_EQ(compact(11, 10), last);
  EXPECT_TRUE(compact(13, 8).empty());

  d.images = {2, 1, {0, 1, 2}, {0, 0}, {9, 11}};
  EXPECT_EQ(compact(10, 8), (std::vector<std::int32_t>{1}));
  EXPECT_TRUE(compact(13, 12).empty());

  EXPECT_THROW(compact(10, 10), std::invalid_argument);
}
#endif

namespace {

constexpr sdnn::Backend kGpuLike{"gpu_like", nullptr, "cuda"};
constexpr sdnn::Backend kNoOptions{"no_options", nullptr};

} // namespace

TEST(SetOption, ParsesInteger) {
  sdnn::InferOptions options;
  sdnn::set_option(kGpuLike, options, "cuda.compact_after", "7");
  EXPECT_EQ(options.cuda.compact_after, 7);
}

TEST(SetOption, UnknownKeyThrows) {
  sdnn::InferOptions options;
  EXPECT_THROW(sdnn::set_option(kGpuLike, options, "cuda.nope", "1"),
               std::invalid_argument);
}

TEST(SetOption, MalformedValueThrows) {
  sdnn::InferOptions options;
  EXPECT_THROW(sdnn::set_option(kGpuLike, options, "cuda.compact_after", "8x"),
               std::invalid_argument);
  EXPECT_THROW(sdnn::set_option(kGpuLike, options, "cuda.compact_after", ""),
               std::invalid_argument);
}

TEST(SetOption, OtherBackendsOptionsThrow) {
  sdnn::InferOptions options;
  EXPECT_THROW(sdnn::set_option(kNoOptions, options, "cuda.compact_after", "8"),
               std::invalid_argument);
  EXPECT_EQ(options.cuda.compact_after, 0);
}

TEST(SetOption, RegisteredCpuBackendsRejectGpuOptions) {
  for (const auto &backend : sdnn::backends()) {
    if (backend.option_group.empty()) {
      sdnn::InferOptions options;
      EXPECT_THROW(
          sdnn::set_option(backend, options, "cuda.compact_after", "8"),
          std::invalid_argument)
          << backend.name;
    }
  }
}

TEST(Backends, PresetNamesAreUnique) {
  const auto all = sdnn::backends();
  for (std::size_t i = 0; i < all.size(); ++i) {
    for (std::size_t j = i + 1; j < all.size(); ++j) {
      EXPECT_NE(all[i].name, all[j].name);
    }
  }
}
