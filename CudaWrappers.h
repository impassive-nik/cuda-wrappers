#ifndef __CUDA_WRAPPERS_H__
#define __CUDA_WRAPPERS_H__

#include "Grid2D.h"

#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cw {

// returns memory size in normalized format, e.g. "512 Bytes", "2.5 Gb" etc.
std::string memsizeToString(size_t bytes);

struct DeviceInfo {
  int id = -1;
  std::string name;
  size_t mem_total = 0;
  size_t mem_shared_per_block = 0;
  size_t warp_size = 0;

  std::string toString() const {
    std::stringstream ss;
    ss << "#" << id << " - " << name << std::endl;
    ss << "  Total global memory: " << memsizeToString(mem_total) << std::endl;
    ss << "  Shared memory per block: " << memsizeToString(mem_shared_per_block)
       << std::endl;
    ss << "  Warp-size: " << memsizeToString(warp_size) << std::endl;
    return ss.str();
  }
};

std::vector<DeviceInfo> getDevices();

std::ostream &operator<<(std::ostream &os, const DeviceInfo &di);

class DeviceMemory {
  size_t size;
  void *ptr;
public:
  DeviceMemory(size_t size);

  void copy_to_device(void *from);
  void copy_from_device(void *to);

  size_t getSize() const { return size; }
  void *getPtr() const { return ptr; }

  ~DeviceMemory();
};

class DeviceGrid2DImpl {
protected:
  DeviceMemory data;
  Grid2DImpl &host_grid;

public:
  DeviceGrid2DImpl(Grid2DImpl &grid) : host_grid(grid), data(grid.data.size()) {}

  void copyToDevice() { data.copy_to_device(host_grid.data.data()); }
  void copyToHost() { data.copy_from_device(host_grid.data.data()); }

  using fun_ptr_t = void (*)(void *);
  void calculate(fun_ptr_t fun_ptr);
};

template<typename elem_t>
class DeviceGrid2D : public DeviceGrid2DImpl {
public:
  DeviceGrid2D(Grid2D<elem_t> &grid) : DeviceGrid2DImpl(grid) {}

  using fun_ptr_t = void (*)(elem_t *);
  void calculate(fun_ptr_t fun_ptr) {
    DeviceGrid2DImpl::calculate((DeviceGrid2DImpl::fun_ptr_t) fun_ptr);
  }
};

} // namespace cw

#endif // __CUDA_WRAPPERS_H__