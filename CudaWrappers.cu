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

void DeviceMemory::copy_to_device(void* from) {
  cudaMemcpy(ptr, from, size, cudaMemcpyHostToDevice);
}

void DeviceMemory::copy_from_device(void *to) {
  cudaMemcpy(to, ptr, size, cudaMemcpyDeviceToHost);
}

DeviceMemory::DeviceMemory(size_t size) : size(size) { cudaMalloc(&ptr, size); }
DeviceMemory::~DeviceMemory() { cudaFree(ptr); }

__global__ void calculateImpl(void *ptr, unsigned row_length, size_t elem_size, void (*fun_ptr)(void *)) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  printf("[%d]", *(int *)ptr);
  fun_ptr(ptr);
  printf("(%d)\n", *(int *)ptr);
}

void DeviceGrid2DImpl::calculate(fun_ptr_t fun_ptr) {
  calculateImpl<<<host_grid.width, host_grid.height>>>(data.getPtr(), 
      host_grid.row_length, 
      host_grid.elem_size, 
      fun_ptr);
}

} // namespace cw