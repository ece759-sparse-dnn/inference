#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace sdnn {

template <typename T> struct NpyArray {
  std::vector<std::size_t> shape;
  std::vector<T> data;
};

template <typename T> NpyArray<T> load_npy(const std::filesystem::path &path);

extern template NpyArray<std::int32_t> load_npy(const std::filesystem::path &);
extern template NpyArray<float> load_npy(const std::filesystem::path &);

} // namespace sdnn
