#include "CudaWrappers.h"

std::vector<cw::DeviceInfo> cw::getDevices() {
  std::vector<cw::DeviceInfo> result;

  int n;
  cudaGetDeviceCount(&n);

  for (int i = 0; i < n; i++) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, i);

    cw::DeviceInfo info;
    info.id   = i;
    info.name = properties.name;
    result.emplace_back(std::move(info));
  }

  return result;
}