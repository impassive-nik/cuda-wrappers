#ifndef __GRID2D_H__
#define __GRID2D_H__

#include "CudaWrappers.h"

#include <stdint.h>
#include <vector>

namespace cw {

struct Pos {
  const uint8_t *base_ptr;
  const size_t elem_size;
  const size_t row_length;
  void *ptr;

  Pos(uint8_t *base_ptr, uint32_t x, uint32_t y, size_t elem_size, size_t row_length):
    base_ptr(base_ptr), ptr(base_ptr + x * elem_size + y * row_length), elem_size(elem_size), row_length(row_length) {}
};
  
inline CUDA_HOSTDEV uint32_t getX(const Pos &p) {
  return ((((uint8_t *)p.ptr - p.base_ptr) % p.row_length) / p.elem_size);
}

inline CUDA_HOSTDEV uint32_t getY(const Pos &p) {
  return ((((uint8_t *)p.ptr - p.base_ptr) / p.row_length) / p.elem_size);
}

template <typename elem_t>
inline CUDA_HOSTDEV elem_t *get(const Pos &p) {
  return (elem_t *) p.ptr;
}

template <typename elem_t>
CUDA_HOSTDEV elem_t *offset(const Pos &p, int32_t dx, int32_t dy) {
  return (elem_t *) (void *)((((uint8_t *) p.ptr) + dx * (int64_t) p.elem_size + dy * (int64_t) p.row_length));
}

struct Grid2DInfo {
  uint32_t elem_size;
  uint32_t width;
  uint32_t height;
  uint32_t row_length;
  uint32_t padding;
  
  Grid2DInfo(uint32_t elem_size, uint32_t width, uint32_t height, uint32_t pad = 1):
    elem_size(elem_size), width(width), height(height), padding(pad) {
      auto row_padding = (pad - ((elem_size * width) % pad)) % pad;
      row_length = elem_size * width + row_padding;
  }

  Pos at(void *base_ptr, uint32_t x, uint32_t y) const {
    return Pos((uint8_t *) base_ptr, x, y, elem_size, row_length);
  }

  size_t size() const {
    return (size_t) row_length * height;
  }
};

struct Grid2DImpl {
  const Grid2DInfo info;
  std::vector<uint8_t> data;

  Grid2DImpl(Grid2DInfo info): info(info), data(info.size()) {
  }

  Grid2DImpl(uint32_t elem_size, uint32_t width, uint32_t height, uint32_t pad = 1):
    Grid2DImpl(Grid2DInfo(elem_size, width, height, pad)) {
  }

  const void *at(uint32_t x, uint32_t y) const {
    return &data[(size_t)info.row_length * y + x * info.elem_size];
  }

  void *at(uint32_t x, uint32_t y) {
    return &data[(size_t) info.row_length * y + x * info.elem_size];
  }

  virtual ~Grid2DImpl() {
  }
};

template <typename elem_t>
struct Grid2D : public Grid2DImpl {
  Grid2D(uint32_t width, uint32_t height, uint32_t pad = 1): Grid2DImpl(Grid2DInfo(sizeof(elem_t), width, height, pad)) {
  }

  Grid2D(Grid2DInfo info): Grid2D(info.width, info.height, info.padding) {
  }

  const elem_t *at(uint32_t x, uint32_t y) const {
    return (const elem_t *) Grid2DImpl::at(x, y);
  }

  elem_t *at(uint32_t x, uint32_t y) {
    return (elem_t *) Grid2DImpl::at(x, y);
  }

  using do_fun_t = void(Pos);
  using update_fun_t = elem_t(Pos);
  
  template<do_fun_t fun_ptr>
  void cellDo(uint32_t border = 0) {
    //TODO: openMP?
    for (unsigned y = border; y < info.height - border; y++)
      for (unsigned x = border; x < info.width - border; x++)
        fun_ptr(info.at(data.data(), x, y));
  }
  
  template<update_fun_t fun_ptr>
  void cellUpdate(uint32_t border = 0) {
    auto data_copy = data;
    
    //TODO: openMP?
    for (unsigned y = border; y < info.height - border; y++)
      for (unsigned x = border; x < info.width - border; x++)
        *get<elem_t>(info.at(data_copy.data(), x, y)) = fun_ptr(info.at(data.data(), x, y));
    
    data = std::move(data_copy);
  }
};

} // namespace cw

#endif // __GRID2D_H__