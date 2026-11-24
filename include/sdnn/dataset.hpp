#pragma once

#include <sdnn/csr.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <vector>

namespace sdnn {

inline constexpr int kBinVersion = 1;

struct Dataset {
  float bias = 0.0f;
  float ymax = 0.0f;
  Csr images;
  CsrLayers layers;
  std::map<int, std::vector<std::int32_t>> categories;
};

Dataset load_dataset(const std::filesystem::path &dir);

} // namespace sdnn
