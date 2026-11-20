#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>

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
    auto result = backend.infer(*dataset, 120);
    ASSERT_EQ(result.size(), truth.size());
    EXPECT_EQ(result, truth);
  } catch (const sdnn::not_implemented &e) {
    GTEST_SKIP() << e.what();
  }
}
