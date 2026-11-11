#include "sdnn/npy.hpp"

#include <bit>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

namespace sdnn {

namespace {

template <typename T> constexpr std::string_view descr();
template <> constexpr std::string_view descr<std::int32_t>() { return "<i4"; }
template <> constexpr std::string_view descr<float>() { return "<f4"; }

std::string header_field(const std::string &header, const char *pattern,
                         const std::filesystem::path &path) {
  std::smatch match;
  if (!std::regex_search(header, match, std::regex(pattern))) {
    throw std::runtime_error(path.string() + ": malformed npy header");
  }
  return match[1];
}

} // namespace

template <typename T> NpyArray<T> load_npy(const std::filesystem::path &path) {
  static_assert(std::endian::native == std::endian::little);

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error(path.string() + ": cannot open");
  }

  char magic[8];
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error(path.string() + ": not an npy file");
  }

  std::uint32_t header_len = 0;
  if (magic[6] == 1) {
    std::uint16_t len = 0;
    in.read(reinterpret_cast<char *>(&len), sizeof(len));
    header_len = len;
  } else {
    in.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
  }
  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  if (!in) {
    throw std::runtime_error(path.string() + ": truncated npy header");
  }

  if (header_field(header, R"('descr':\s*'([^']*)')", path) != descr<T>()) {
    throw std::runtime_error(path.string() + ": expected dtype " +
                             std::string(descr<T>()));
  }
  if (header_field(header, R"('fortran_order':\s*(\w+))", path) != "False") {
    throw std::runtime_error(path.string() + ": fortran order unsupported");
  }

  NpyArray<T> array;
  std::size_t count = 1;
  const std::string dims =
      header_field(header, R"('shape':\s*\(([^)]*)\))", path);
  const std::regex dim(R"(\d+)");
  for (auto it = std::sregex_iterator(dims.begin(), dims.end(), dim);
       it != std::sregex_iterator(); ++it) {
    array.shape.push_back(std::stoull(it->str()));
    count *= array.shape.back();
  }

  array.data.resize(count);
  in.read(reinterpret_cast<char *>(array.data.data()),
          static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) {
    throw std::runtime_error(path.string() + ": truncated npy data");
  }
  return array;
}

template NpyArray<std::int32_t> load_npy(const std::filesystem::path &);
template NpyArray<float> load_npy(const std::filesystem::path &);

} // namespace sdnn
