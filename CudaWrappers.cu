#include "CudaWrappers.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace cw {

std::string memsizeToString(size_t bytes) {
  std::array<const char *, 4> names = {"KB", "MB", "GB", "TB"};
  const char *prefix = "Bytes";

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

} // namespace cw