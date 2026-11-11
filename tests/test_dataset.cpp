#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

#include "sdnn/dataset.hpp"

TEST(Dataset, Load1024) {
  const char *env = std::getenv("SDNN_DATA_DIR");
  const std::filesystem::path root = env ? env : SDNN_DEFAULT_DATA_DIR;
  const auto dir = root / "bin" / "neuron1024";
  if (!std::filesystem::exists(dir / ".done")) {
    GTEST_SKIP() << "no converted dataset at " << dir;
  }

  const sdnn::Dataset d = sdnn::load_dataset(dir);
  EXPECT_EQ(d.neurons, 1024);
  EXPECT_EQ(d.images.indptr.size(), static_cast<std::size_t>(d.images.rows) + 1);
  EXPECT_EQ(d.layer_indptr(d.num_layers - 1).back(), d.nnz_per_layer);
  EXPECT_TRUE(d.categories.contains(120));
}
