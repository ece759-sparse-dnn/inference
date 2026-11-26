// Backend correctness on small synthetic networks built in memory.
//
// Unlike BackendTest.MatchesTruth, these need no dataset, so they run in CI
// (CPU backends) and in scripts/slurm/test.sbatch (every backend, CUDA
// included). Each registered backend is compared against reference(), a plain
// dense implementation of the layer rule, at every depth from 0 to the
// network's layer count.
//
// The random networks use small integer weights, an integer bias and a small
// ymax, so every activation is a small integer and float arithmetic is exact.
// Backends that sum in different orders (scatter on the CPU, gather on the
// GPU) must then agree with the reference exactly.
//
// Constraints every network here respects:
// - Each layer has the same nonzero count in every row AND every column: the
//   CUDA backend stores layers as ELL of the transpose, which needs constant
//   nonzeros per column (ell::from_transpose throws otherwise).
// - The bias is negative, as in the challenge. Backends add it only to
//   positive sums; the reference adds it to every nonzero sum. The two rules
//   agree only when bias <= 0.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sdnn/backend.hpp"

namespace {

using Rows = std::vector<std::int32_t>;

// Row-major dense matrix, for describing networks and for the reference.
struct Dense {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::vector<float> values;

  Dense(std::int32_t r, std::int32_t c)
      : rows(r), cols(c), values(static_cast<std::size_t>(r) * c, 0.0f) {}

  Dense(std::initializer_list<std::initializer_list<float>> init)
      : Dense(static_cast<std::int32_t>(init.size()),
              init.size() == 0
                  ? 0
                  : static_cast<std::int32_t>(init.begin()->size())) {
    std::int32_t r = 0;
    for (const auto &row : init) {
      std::int32_t c = 0;
      for (const float v : row) {
        at(r, c++) = v;
      }
      ++r;
    }
  }

  float &at(std::int32_t r, std::int32_t c) {
    return values[static_cast<std::size_t>(r) * cols + c];
  }
  float at(std::int32_t r, std::int32_t c) const {
    return values[static_cast<std::size_t>(r) * cols + c];
  }
};

Dense scaled(Dense m, float factor) {
  for (auto &v : m.values) {
    v *= factor;
  }
  return m;
}

sdnn::Csr to_csr(const Dense &m) {
  sdnn::Csr out{m.rows, m.cols, {0}, {}, {}};
  for (std::int32_t r = 0; r < m.rows; ++r) {
    for (std::int32_t c = 0; c < m.cols; ++c) {
      if (m.at(r, c) != 0.0f) {
        out.indices.push_back(c);
        out.values.push_back(m.at(r, c));
      }
    }
    out.indptr.push_back(static_cast<std::int32_t>(out.values.size()));
  }
  return out;
}

sdnn::Dataset make_dataset(float bias, float ymax, const Dense &images,
                           const std::vector<Dense> &layers) {
  sdnn::Dataset d;
  d.bias = bias;
  d.ymax = ymax;
  d.images = to_csr(images);

  auto &w = d.layers;
  w.neurons = images.cols;
  w.num_layers = static_cast<std::int32_t>(layers.size());
  for (std::size_t l = 0; l < layers.size(); ++l) {
    if (layers[l].rows != w.neurons || layers[l].cols != w.neurons) {
      throw std::logic_error("each layer must be neurons x neurons");
    }
    const auto csr = to_csr(layers[l]);
    const auto nnz = static_cast<std::int32_t>(csr.values.size());
    if (l == 0) {
      w.nnz_per_layer = nnz;
    } else if (nnz != w.nnz_per_layer) {
      throw std::logic_error("every layer needs the same nonzero count");
    }
    w.indptr.insert(w.indptr.end(), csr.indptr.begin(), csr.indptr.end());
    w.indices.insert(w.indices.end(), csr.indices.begin(), csr.indices.end());
    w.values.insert(w.values.end(), csr.values.begin(), csr.values.end());
  }
  return d;
}

Rows active_rows(const Dense &y) {
  Rows rows;
  for (std::int32_t r = 0; r < y.rows; ++r) {
    for (std::int32_t c = 0; c < y.cols; ++c) {
      if (y.at(r, c) != 0.0f) {
        rows.push_back(r);
        break;
      }
    }
  }
  return rows;
}

struct Reference {
  std::vector<Rows> active;  // active[l] = active rows after l layers
  long clamped = 0;          // entries that hit ymax, over all layers
};

// The layer rule straight from the definition (python reference.py):
// Z = Y W; add the bias to the nonzero entries of Z; clamp to [0, ymax].
Reference reference(const sdnn::Dataset &d) {
  const auto n = d.layers.neurons;
  Dense y(d.images.rows, n);
  for (std::int32_t r = 0; r < d.images.rows; ++r) {
    for (auto p = d.images.indptr[r]; p < d.images.indptr[r + 1]; ++p) {
      y.at(r, d.images.indices[p]) = d.images.values[p];
    }
  }

  Reference ref;
  ref.active.push_back(active_rows(y));
  for (std::int32_t l = 0; l < d.layers.num_layers; ++l) {
    const auto layer = d.layers.layer(l);
    Dense w(n, n);
    for (std::int32_t k = 0; k < n; ++k) {
      for (auto q = layer.indptr[k]; q < layer.indptr[k + 1]; ++q) {
        w.at(k, layer.indices[q]) = layer.values[q];
      }
    }

    Dense z(y.rows, n);
    for (std::int32_t i = 0; i < y.rows; ++i) {
      for (std::int32_t k = 0; k < n; ++k) {
        if (y.at(i, k) == 0.0f) {
          continue;
        }
        for (std::int32_t j = 0; j < n; ++j) {
          z.at(i, j) += y.at(i, k) * w.at(k, j);
        }
      }
    }
    for (auto &v : z.values) {
      if (v != 0.0f) {
        v += d.bias;
        if (v > d.ymax) {
          v = d.ymax;
          ++ref.clamped;
        }
        if (v < 0.0f) {
          v = 0.0f;
        }
      }
    }
    y = std::move(z);
    ref.active.push_back(active_rows(y));
  }
  return ref;
}

// From python/tests/test_reference.py: 4 neurons, 2 edges per row and
// column, 3 images (the middle one has no pixels).
sdnn::Dataset ring_network(float bias = -0.6f) {
  const Dense w0{{1, 1, 0, 0}, {0, 1, 1, 0}, {0, 0, 1, 1}, {1, 0, 0, 1}};
  const Dense images{{1, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 1, 1}};
  return make_dataset(bias, 32.0f, images, {w0, scaled(w0, 0.25f)});
}

} // namespace

// Outside the anonymous namespace: GoogleTest stores test parameters in
// classes with external linkage (otherwise GCC warns -Wsubobject-linkage).
namespace synthetic {

struct RandomCase {
  const char *name;
  std::int32_t neurons;
  std::int32_t k;  // edges out of and into every neuron
  int layers;
  std::int32_t images;
  int density_percent;   // chance that a pixel is 1
  int negative_percent;  // chance that a weight is -1 (otherwise +1)
  float bias;            // integer-valued and negative
  float ymax;            // integer-valued
  std::uint32_t seed;
};

} // namespace synthetic

namespace {

using synthetic::RandomCase;

// More than 8 layers so the cuda_compact8 preset is exercised too.
// 300 neurons: more than one 256-thread CUDA block, and not a multiple of it.
// k = 32: the width of the real challenge networks, so kernels specialized
// for it are covered without the dataset.
//
// Density, weight mix, bias and ymax were tuned per width so images keep
// dying across many layers instead of settling in the first few (checked by
// RandomNetworksAreMeaningful). Retune if you change them.
const RandomCase kRandomCases[] = {
    // name               n    k  layers images dens neg  bias  ymax seed
    {"n64_seed1",         64,  4, 20,    256,   30,  25,  -1.0f, 3.0f, 1},
    {"n64_seed2",         64,  4, 20,    256,   30,  25,  -1.0f, 3.0f, 2},
    {"n300_two_blocks",   300, 4, 12,    64,    30,  25,  -1.0f, 3.0f, 3},
    {"n96_width32",       96,  32, 20,   256,   20,  50,  -3.0f, 2.0f, 11},
};

// Row r of each layer connects to perm[(r + shift) % n] for k distinct
// shifts, so every row and every column has exactly k nonzeros. Each weight
// is -1 with probability negative_percent, else +1, so sums can cancel. With
// an integer bias and ymax, every activation is an integer in [0, ymax].
//
// Uses the raw mt19937 stream (specified exactly by the standard) instead of
// std::*_distribution, whose output differs between standard libraries.
sdnn::Dataset random_regular(const RandomCase &c) {
  std::mt19937 rng(c.seed);
  const auto below = [&rng](std::int32_t n) {
    return static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n));
  };
  const auto shuffled = [&below](std::int32_t n) {
    std::vector<std::int32_t> v(static_cast<std::size_t>(n));
    std::iota(v.begin(), v.end(), 0);
    for (std::int32_t i = n - 1; i > 0; --i) {
      std::swap(v[i], v[below(i + 1)]);
    }
    return v;
  };

  std::vector<Dense> layers;
  for (int l = 0; l < c.layers; ++l) {
    const auto shifts = shuffled(c.neurons);  // the first k are distinct
    const auto perm = shuffled(c.neurons);
    Dense w(c.neurons, c.neurons);
    for (std::int32_t r = 0; r < c.neurons; ++r) {
      for (std::int32_t s = 0; s < c.k; ++s) {
        w.at(r, perm[(r + shifts[s]) % c.neurons]) =
            below(100) < c.negative_percent ? -1.0f : 1.0f;
      }
    }
    layers.push_back(std::move(w));
  }

  Dense images(c.images, c.neurons);
  for (auto &v : images.values) {
    v = below(100) < c.density_percent ? 1.0f : 0.0f;
  }
  return make_dataset(c.bias, c.ymax, images, layers);
}

// Checks backend results at every depth against expected[depth]. Skips
// depths a compaction preset can't run (it must compact before the last
// layer), and the whole test if the backend is a stub.
void expect_all_depths(const sdnn::Backend &backend, const sdnn::Dataset &d,
                       const std::vector<Rows> &expected) {
  const int compact_after = backend.preset.cuda.compact_after;
  int checked = 0;
  for (int layers = 0; layers < static_cast<int>(expected.size()); ++layers) {
    if (compact_after > 0 && compact_after >= layers) {
      continue;
    }
    SCOPED_TRACE("layers = " + std::to_string(layers));
    try {
      EXPECT_EQ(backend.infer(d, layers, backend.preset), expected[layers]);
    } catch (const sdnn::not_implemented &e) {
      GTEST_SKIP() << e.what();
    }
    ++checked;
  }
  if (checked == 0) {
    GTEST_SKIP() << backend.name << " needs more than " << compact_after
                 << " layers";
  }
}

std::string backend_name(const sdnn::Backend &backend) {
  return std::string(backend.name);
}

} // namespace

// ---------------------------------------------------------------------------
// The test networks themselves: regular, and able to catch bugs.
// ---------------------------------------------------------------------------

TEST(SyntheticNetworks, ReferenceMatchesHandComputedRing) {
  // Image 2 survives both layers; image 0 dies in the second; image 1 is
  // empty. Values from python/tests/test_reference.py.
  const std::vector<Rows> expected{{0, 2}, {0, 2}, {2}};
  EXPECT_EQ(reference(ring_network()).active, expected);
}

TEST(SyntheticNetworks, RandomNetworksAreRegular) {
  for (const auto &c : kRandomCases) {
    SCOPED_TRACE(c.name);
    const auto d = random_regular(c);
    for (std::int32_t l = 0; l < d.layers.num_layers; ++l) {
      const auto layer = d.layers.layer(l);
      std::vector<std::int32_t> per_col(c.neurons, 0);
      for (std::int32_t r = 0; r < c.neurons; ++r) {
        ASSERT_EQ(layer.indptr[r + 1] - layer.indptr[r], c.k) << "row " << r;
        for (auto q = layer.indptr[r]; q < layer.indptr[r + 1]; ++q) {
          ++per_col[layer.indices[q]];
        }
      }
      for (std::int32_t col = 0; col < c.neurons; ++col) {
        ASSERT_EQ(per_col[col], c.k) << "layer " << l << " column " << col;
      }
    }
  }
}

// A test that passes for "everything dies" or "nothing changes" proves
// little. Each random network must have survivors and casualties at the last
// depth, an active set that still changes after the first few layers (so an
// off-by-one in the layer count fails), and values that hit ymax.
TEST(SyntheticNetworks, RandomNetworksAreMeaningful) {
  for (const auto &c : kRandomCases) {
    SCOPED_TRACE(c.name);
    const auto ref = reference(random_regular(c));
    const auto &last = ref.active.back();
    EXPECT_FALSE(last.empty()) << "every image died";
    EXPECT_LT(last.size(), static_cast<std::size_t>(c.images))
        << "no image died";

    int late_changes = 0;
    for (std::size_t l = 4; l < ref.active.size(); ++l) {
      late_changes += ref.active[l] != ref.active[l - 1];
    }
    EXPECT_GT(late_changes, 0) << "active set stops changing after 3 layers";
    EXPECT_GT(ref.clamped, 0) << "ymax is never reached";
  }
}

#ifdef CUDA_ENABLED
TEST(SyntheticNetworks, CudaRejectsIrregularColumns) {
  // Every row has 2 nonzeros, but columns have 3, 2, 2 and 1.
  const Dense w{{1, 1, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {0, 0, 1, 1}};
  const Dense images{{1, 0, 0, 0}};
  const auto d = make_dataset(-0.5f, 32.0f, images, {w});
  EXPECT_THROW(sdnn::infer_cuda(d, 1, {}), std::invalid_argument);
}
#endif

// ---------------------------------------------------------------------------
// Every registered backend on the hand-written networks.
// ---------------------------------------------------------------------------

class SyntheticBackend : public testing::TestWithParam<sdnn::Backend> {};

INSTANTIATE_TEST_SUITE_P(
    AllBackends, SyntheticBackend, testing::ValuesIn(sdnn::backends()),
    [](const testing::TestParamInfo<sdnn::Backend> &info) {
      return backend_name(info.param);
    });

TEST_P(SyntheticBackend, RingNetwork) {
  expect_all_depths(GetParam(), ring_network(), {{0, 2}, {0, 2}, {2}});
}

TEST_P(SyntheticBackend, AllImagesDie) {
  expect_all_depths(GetParam(), ring_network(-100.0f), {{0, 2}, {}, {}});
}

TEST_P(SyntheticBackend, NoPixelsNeverActive) {
  const Dense w0{{1, 1, 0, 0}, {0, 1, 1, 0}, {0, 0, 1, 1}, {1, 0, 0, 1}};
  const auto d = make_dataset(-0.5f, 32.0f, Dense(3, 4), {w0, w0});
  expect_all_depths(GetParam(), d, {{}, {}, {}});
}

// ---------------------------------------------------------------------------
// Every registered backend on every random network, against the reference.
// ---------------------------------------------------------------------------

class SyntheticRandom
    : public testing::TestWithParam<std::tuple<sdnn::Backend, RandomCase>> {};

INSTANTIATE_TEST_SUITE_P(
    AllBackends, SyntheticRandom,
    testing::Combine(testing::ValuesIn(sdnn::backends()),
                     testing::ValuesIn(kRandomCases)),
    [](const testing::TestParamInfo<SyntheticRandom::ParamType> &info) {
      return backend_name(std::get<0>(info.param)) + "_" +
             std::get<1>(info.param).name;
    });

TEST_P(SyntheticRandom, MatchesReference) {
  const auto &[backend, c] = GetParam();
  const auto d = random_regular(c);
  expect_all_depths(backend, d, reference(d).active);
}
