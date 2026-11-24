#pragma once

#include "check.cuh"

#include <cuda/std/span>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

namespace sdnn::cuda {

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t size) : size_{size} {
    check(cudaMalloc(&dev_ptr_, sizeof(T) * size_));
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  auto operator=(const DeviceBuffer &) -> DeviceBuffer & = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : dev_ptr_{std::exchange(other.dev_ptr_, nullptr)},
        size_{std::exchange(other.size_, 0)} {}

  auto operator=(DeviceBuffer &&other) noexcept -> DeviceBuffer & {
    if (this != &other) {
      cudaFree(dev_ptr_);
      dev_ptr_ = std::exchange(other.dev_ptr_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  ~DeviceBuffer() { cudaFree(dev_ptr_); }

  auto upload(std::span<const T> data) -> void {
    if (data.size() != size_) {
      throw std::invalid_argument("upload size must match buffer size");
    }
    check(cudaMemcpy(dev_ptr_, data.data(), sizeof(T) * size_,
                     cudaMemcpyHostToDevice));
  }

  auto download(std::span<T> data) const -> void {
    if (data.size() != size_) {
      throw std::invalid_argument("download size must match buffer size");
    }
    check(cudaMemcpy(data.data(), dev_ptr_, sizeof(T) * size_,
                     cudaMemcpyDeviceToHost));
  }

  auto data() -> T * { return dev_ptr_; }
  auto data() const -> const T * { return dev_ptr_; }

  auto size() const -> std::size_t { return size_; }

  auto span() -> ::cuda::std::span<T> { return {dev_ptr_, size_}; }
  auto span() const -> ::cuda::std::span<const T> { return {dev_ptr_, size_}; }

private:
  T *dev_ptr_ = nullptr;
  std::size_t size_ = 0;
};

} // namespace sdnn::cuda
