#ifndef __CUDA_WRAPPERS_H__
#define __CUDA_WRAPPERS_H__

#include "Grid2D.h"

#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace cw {

// returns memory size in normalized format, e.g. "512 Bytes", "2.5 Gb" etc.
std::string memsizeToString(size_t bytes);

// stores information about a GPU device
struct DeviceInfo {
  int id = -1;
  std::string name;
  size_t mem_total = 0;
  size_t mem_shared_per_block = 0;
  size_t warp_size = 0;

  std::string toString() const;
};

std::ostream &operator<<(std::ostream &os, const DeviceInfo &di);

// returns vector of GPU devices' information
std::vector<DeviceInfo> getDevices();

// wrapper for an allocated GPU memory
class DeviceMemory {
  size_t size;
  void *ptr;

public:
  DeviceMemory(size_t size);
  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory &operator=(const DeviceMemory &) = delete;

  void copy_from(const void *from);
  void copy_from(const DeviceMemory &from);
  void copy_to(void *to) const;

  size_t getSize() const { return size; }
  void *getPtr() const { return ptr; }

  ~DeviceMemory();
};

} // namespace cw

#endif // __CUDA_WRAPPERS_H__