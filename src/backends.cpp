#include "sdnn/backend.hpp"

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace sdnn {

namespace {

int parse_int(std::string_view key, std::string_view value) {
  int result = 0;
  const auto *end = value.data() + value.size();
  const auto [ptr, ec] = std::from_chars(value.data(), end, result);
  if (ec != std::errc{} || ptr != end) {
    throw std::invalid_argument("option " + std::string(key) +
                                " expects an integer, got '" +
                                std::string(value) + "'");
  }
  return result;
}

} // namespace

std::span<const Backend> backends() {
  static constexpr Backend all[] = {
      {"serial", infer_serial},
      {"openmp", infer_openmp},
#ifdef CUDA_ENABLED
      {"cuda", infer_cuda, "cuda"},
      {"cuda_compact8", infer_cuda, "cuda", {.cuda = {.compact_after = 8}}},
#endif
  };
  return all;
}

void set_option(const Backend &backend, InferOptions &options,
                std::string_view key, std::string_view value) {
  const auto group = key.substr(0, key.find('.'));
  if (backend.option_group.empty() || group != backend.option_group) {
    throw std::invalid_argument("option '" + std::string(key) +
                                "' does not apply to backend '" +
                                std::string(backend.name) + "'");
  }

  if (key == "cuda.compact_after") {
    options.cuda.compact_after = parse_int(key, value);
  } else {
    throw std::invalid_argument("unknown option '" + std::string(key) + "'");
  }
}

} // namespace sdnn
