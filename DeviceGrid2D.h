#ifndef __DEVICE_GRID_2D_H__
#define __DEVICE_GRID_2D_H__

#include "CudaWrappers.h"
#include "Grid2D.h"

#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace cw {

// prints to stderr and returns true if cudaGetLastError() contains an error
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
bool checkLast(const char* const file, const int line);

#ifdef CUDA_DEFINED

template<typename elem_t, void (*fun_ptr)(Pos)>
__global__ void cellDoImpl(Pos pos) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  pos.ptr = offset<void>(pos, x, y);
  fun_ptr(pos);
}

template<typename elem_t, elem_t (*fun_ptr)(Pos)>
__global__ void cellUpdateImpl(Pos pos, Pos pos_to) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  pos.ptr = offset<void>(pos, x, y);
  auto ptr_to = offset<elem_t>(pos_to, x, y);
  *ptr_to = fun_ptr(pos);
}

#endif // CUDA_DEFINED

class DeviceGrid2DImpl {
protected:
  DeviceMemory data;

public:
  const Grid2DInfo info;

  DeviceGrid2DImpl(Grid2DInfo info) : info(info), data(info.size()) {}
  DeviceGrid2DImpl(uint32_t elem_size, uint32_t width, uint32_t height, uint32_t pad = 1): DeviceGrid2DImpl(Grid2DInfo(elem_size, width, height, pad)) {}
  DeviceGrid2DImpl(const Grid2DImpl &grid) : DeviceGrid2DImpl(grid.info) {}
  DeviceGrid2DImpl(const DeviceGrid2DImpl &grid) : DeviceGrid2DImpl(grid.info) {}

  void copyFrom(const Grid2DImpl &host_grid) { data.copy_from(host_grid.data.data()); }
  void copyTo(Grid2DImpl &host_grid) { data.copy_to(host_grid.data.data()); }

  virtual ~DeviceGrid2DImpl() {
  }
};

#ifdef CUDA_DEFINED

template<typename elem_t>
class DeviceGrid2D : public DeviceGrid2DImpl {
  std::unique_ptr<DeviceMemory> data_copy;
public:
  DeviceGrid2D(uint32_t width, uint32_t height, uint32_t pad = 1): DeviceGrid2DImpl(Grid2DInfo(sizeof(elem_t), width, height, pad)) {}
  DeviceGrid2D(const Grid2DInfo &info): DeviceGrid2D(info.width, info.height, info.padding) {}
  DeviceGrid2D(const Grid2DImpl &grid) : DeviceGrid2D(grid.info) {}
  DeviceGrid2D(const DeviceGrid2DImpl &grid) : DeviceGrid2D(grid.info) {}

  using do_fun_t = typename Grid2D<elem_t>::do_fun_t;
  using update_fun_t = typename Grid2D<elem_t>::update_fun_t;
  
  template<do_fun_t fun_ptr>
  void cellDo(uint32_t border = 0) {
    auto width = info.width - 2 * border;
    auto height =  info.height - 2 * border;

    auto pos = info.at(data.getPtr(), border, border);
    pos.base_ptr = (uint8_t *)pos.ptr;
    cellDoImpl<elem_t, fun_ptr> <<<dim3(width, height), 1>>>(pos);
    CHECK_LAST_CUDA_ERROR();
  }
  
  template<update_fun_t fun_ptr>
  void cellUpdate(uint32_t border = 0) {
    if (!data_copy)
      data_copy.reset(new DeviceMemory(data.getSize()));
    data_copy->copy_from(data);

    auto width = info.width - 2 * border;
    auto height = info.height - 2 * border;

    auto pos = info.at(data.getPtr(), border, border);
    pos.base_ptr = (uint8_t *)pos.ptr;

    auto pos_to = info.at(data_copy->getPtr(), border, border);
    pos_to.base_ptr = (uint8_t *)pos_to.ptr;
    
    cellUpdateImpl<elem_t, fun_ptr> <<<dim3(width, height), 1>>>(pos, pos_to);
    CHECK_LAST_CUDA_ERROR();
    
    data.copy_from(*data_copy);
  }
};

#endif // CUDA_DEFINED

} // namespace cw

#endif // __DEVICE_GRID_2D_H__