#include "CudaWrappers.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace cw {

bool checkLast(const char* const file, const int line) {
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line
              << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    return true;
  }
  return false;
}

std::string DeviceInfo::toString() const {
  std::stringstream ss;
  ss << "#" << id << " - " << name << std::endl;
  ss << "  Total global memory: " << memsizeToString(mem_total) << std::endl;
  ss << "  Shared memory per block: " << memsizeToString(mem_shared_per_block)
      << std::endl;
  ss << "  Warp-size: " << memsizeToString(warp_size) << std::endl;
  return ss.str();
}

std::string memsizeToString(size_t bytes) {
  std::array<const char *, 5> names = {"Bytes", "KB", "MB", "GB", "TB"};
  const char *prefix = names[0];

  size_t integer_part = bytes;
  size_t fraction = 0;

  for (int i = 1; integer_part >= 1024 && i < names.size(); i++) {
    fraction = integer_part % 1024;
    integer_part /= 1024;
    prefix = names[i];
  }

  std::stringstream ss;
  ss << integer_part;

  auto decimal_fraction = (fraction * 10 / 1024);
  if (fraction > 0)
    ss << "." << decimal_fraction;
  ss << " " << prefix;

  return ss.str();
}

std::vector<DeviceInfo> getDevices() {
  std::vector<DeviceInfo> result;

  int n;
  cudaGetDeviceCount(&n);

  for (int i = 0; i < n; i++) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, i);

    DeviceInfo info;
    info.id = i;
    info.name = properties.name;
    info.mem_total = properties.totalGlobalMem;
    info.mem_shared_per_block = properties.sharedMemPerBlock;
    info.warp_size = properties.warpSize;

    result.emplace_back(std::move(info));
  }

  return result;
}

std::ostream &operator<<(std::ostream &os, const DeviceInfo &di) {
  os << di.toString();
  return os;
}

void DeviceMemory::copy_from(const DeviceMemory &from) {
  cudaMemcpy(ptr, from.ptr, size, cudaMemcpyDeviceToDevice);
}

void DeviceMemory::copy_from(const void* from) {
  cudaMemcpy(ptr, from, size, cudaMemcpyHostToDevice);
}

void DeviceMemory::copy_to(void *to) const {
  cudaMemcpy(to, ptr, size, cudaMemcpyDeviceToHost);
}

DeviceMemory::DeviceMemory(size_t size) : size(size) { cudaMalloc(&ptr, size); }
DeviceMemory::~DeviceMemory() { cudaFree(ptr); }

} // namespace cw