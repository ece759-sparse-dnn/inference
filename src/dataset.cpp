#include "sdnn/dataset.hpp"

#include <fstream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>

#include "sdnn/npy.hpp"

namespace sdnn {

namespace {

std::string meta_value(const std::string &meta, const std::string &key,
                       const std::string &value_pattern) {
  std::smatch match;
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(" + value_pattern + ")");
  if (!std::regex_search(meta, match, pattern)) {
    throw std::runtime_error("meta.json: missing " + key);
  }
  return match[1];
}

double meta_number(const std::string &meta, const std::string &key) {
  return std::stod(meta_value(meta, key, R"([-+0-9.eE]+)"));
}

std::vector<int> meta_ints(const std::string &meta, const std::string &key) {
  const std::string list = meta_value(meta, key, R"(\[[^\]]*\])");
  const std::regex number(R"(\d+)");
  std::vector<int> values;
  for (auto it = std::sregex_iterator(list.begin(), list.end(), number);
       it != std::sregex_iterator(); ++it) {
    values.push_back(std::stoi(it->str()));
  }
  return values;
}

template <typename T>
std::vector<T> load_shaped(const std::filesystem::path &path,
                           const std::vector<std::size_t> &shape) {
  NpyArray<T> array = load_npy<T>(path);
  if (array.shape != shape) {
    throw std::runtime_error(path.string() + ": unexpected shape");
  }
  return std::move(array.data);
}

} // namespace

Dataset load_dataset(const std::filesystem::path &dir) {
  if (!std::filesystem::exists(dir / ".done")) {
    throw std::runtime_error(dir.string() + ": no converted dataset");
  }
  std::ifstream meta_file(dir / "meta.json");
  const std::string meta{std::istreambuf_iterator<char>(meta_file), {}};
  if (static_cast<int>(meta_number(meta, "version")) != kBinVersion) {
    throw std::runtime_error(dir.string() + ": unsupported bin version");
  }

  Dataset d;
  d.neurons = static_cast<std::int32_t>(meta_number(meta, "neurons"));
  d.num_layers = static_cast<std::int32_t>(meta_number(meta, "layers"));
  d.nnz_per_layer = static_cast<std::int32_t>(meta_number(meta, "nnz_per_layer"));
  d.bias = static_cast<float>(meta_number(meta, "bias"));
  d.ymax = static_cast<float>(meta_number(meta, "ymax"));

  const std::size_t layers = d.num_layers;
  const std::size_t neurons = d.neurons;
  const std::size_t nnz = d.nnz_per_layer;
  d.weights_indptr =
      load_shaped<std::int32_t>(dir / "weights_indptr.npy", {layers, neurons + 1});
  d.weights_indices =
      load_shaped<std::int32_t>(dir / "weights_indices.npy", {layers, nnz});
  d.weights_values = load_shaped<float>(dir / "weights_values.npy", {layers, nnz});

  d.images.rows = static_cast<std::int32_t>(meta_number(meta, "images"));
  d.images.cols = d.neurons;
  d.images.indptr = load_shaped<std::int32_t>(
      dir / "images_indptr.npy", {static_cast<std::size_t>(d.images.rows) + 1});
  const std::size_t image_nnz = d.images.indptr.back();
  d.images.indices =
      load_shaped<std::int32_t>(dir / "images_indices.npy", {image_nnz});
  d.images.values = load_shaped<float>(dir / "images_values.npy", {image_nnz});

  for (const int depth : meta_ints(meta, "category_depths")) {
    const auto path = dir / ("categories_l" + std::to_string(depth) + ".npy");
    d.categories[depth] = load_npy<std::int32_t>(path).data;
  }
  return d;
}

} // namespace sdnn
