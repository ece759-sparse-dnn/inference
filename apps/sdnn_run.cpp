#include <omp.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "sdnn/backend.hpp"
#include "sdnn/dataset.hpp"

namespace {

constexpr std::string_view kCsvHeader =
    "backend,neurons,layers,images,threads,seconds,edges_per_second,pass";

struct Options {
  std::string backend = "serial";
  int neurons = 1024;
  int layers = 120;
  int compact_after = 0;
  std::filesystem::path data_dir;
  std::filesystem::path csv;
};

[[noreturn]] void usage(const char *prog) {
  std::cerr << "usage: " << prog
            << " [--backend NAME] [--size N] [--layers L] [--data-dir DIR]"
               " [--csv FILE]"
#ifdef CUDA_ENABLED
               " [--compact-after K]"
#endif
               "\nbackends:";
  for (const auto &backend : sdnn::backends()) {
    std::cerr << ' ' << backend.name;
  }
  std::cerr << '\n';
  std::exit(2);
}

Options parse(int argc, char **argv) {
  Options opts;
  const char *env = std::getenv("SDNN_DATA_DIR");
  opts.data_dir = env ? env : SDNN_DEFAULT_DATA_DIR;
  for (int i = 1; i < argc; i += 2) {
    const std::string_view flag = argv[i];
    if (i + 1 >= argc) {
      usage(argv[0]);
    }
    const std::string value = argv[i + 1];
    if (flag == "--backend") {
      opts.backend = value;
    } else if (flag == "--size") {
      opts.neurons = std::stoi(value);
    } else if (flag == "--layers") {
      opts.layers = std::stoi(value);
#ifdef CUDA_ENABLED
    } else if (flag == "--compact-after") {
      const char *end = value.data() + value.size();
      const auto [ptr, ec] = std::from_chars(value.data(), end, opts.compact_after);
      if (ec != std::errc{} || ptr != end || opts.compact_after < 1) {
        usage(argv[0]);
      }
#endif
    } else if (flag == "--data-dir") {
      opts.data_dir = value;
    } else if (flag == "--csv") {
      opts.csv = value;
    } else {
      usage(argv[0]);
    }
  }
  if (opts.compact_after &&
      (opts.backend != "cuda" || opts.compact_after >= opts.layers)) {
    usage(argv[0]);
  }
  return opts;
}

std::vector<std::int32_t> run(const sdnn::Backend &backend,
                              const sdnn::Dataset &dataset, const Options &opts) {
#ifdef CUDA_ENABLED
  if (opts.compact_after) {
    return sdnn::infer_cuda_compact(dataset, opts.layers, opts.compact_after);
  }
#endif
  return backend.infer(dataset, opts.layers);
}

} // namespace

int main(int argc, char **argv) {
  const Options opts = parse(argc, argv);
  const auto backends = sdnn::backends();
  const auto backend = std::ranges::find(
      backends, std::string_view(opts.backend), &sdnn::Backend::name);
  if (backend == backends.end()) {
    usage(argv[0]);
  }

  try {
    const auto dir =
        opts.data_dir / "bin" / ("neuron" + std::to_string(opts.neurons));

    const sdnn::Dataset dataset = sdnn::load_dataset(dir);

    const auto truth = dataset.categories.find(opts.layers);
    if (truth == dataset.categories.end() || opts.layers > dataset.layers.num_layers) {
      throw std::runtime_error("no truth categories for " +
                               std::to_string(opts.layers) + " layers");
    }

    const auto start = std::chrono::steady_clock::now();

    const auto rows = run(*backend, dataset, opts);

    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;

    const bool pass = rows == truth->second;
    const double edges = static_cast<double>(dataset.images.rows) *
                         dataset.layers.nnz_per_layer * opts.layers;

    std::ostringstream row;
    row << backend->name;
    if (opts.compact_after) {
      row << "_compact" << opts.compact_after;
    }
    row << ',' << dataset.layers.neurons << ',' << opts.layers << ','
        << dataset.images.rows << ',' << omp_get_max_threads() << ','
        << elapsed.count() << ',' << edges / elapsed.count() << ','
        << (pass ? "true" : "false");

    std::cout << kCsvHeader << '\n' << row.str() << '\n';
    if (!opts.csv.empty()) {
      const bool fresh = !std::filesystem::exists(opts.csv) ||
                         std::filesystem::file_size(opts.csv) == 0;
      std::ofstream out(opts.csv, std::ios::app);
      if (fresh) {
        out << kCsvHeader << '\n';
      }
      out << row.str() << '\n';
    }
    return pass ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
