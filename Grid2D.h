#ifndef __GRID2D_H__
#define __GRID2D_H__

#include "CudaWrappers.h"

#include <stdint.h>
#include <vector>

namespace cw {

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

  struct Pos {
    const uint8_t *base_ptr;
    const size_t elem_size;
    const size_t row_length;
    void *ptr;

    Pos(uint8_t *base_ptr, uint32_t x, uint32_t y, size_t elem_size, size_t row_length):
      base_ptr(base_ptr), ptr(base_ptr + x * elem_size + y * row_length), elem_size(elem_size), row_length(row_length) {}

    #define GET_POS_X(pos) ((((uint8_t *)(pos).ptr - (pos).base_ptr) % (pos).row_length) / (pos).elem_size)
    uint32_t getX() const {
      return GET_POS_X(*this);
    }

    #define GET_POS_Y(pos) ((((uint8_t *)(pos).ptr - (pos).base_ptr) / (pos).row_length) / (pos).elem_size)
    uint32_t getY() const {
      return GET_POS_Y(*this);
    }

    #define GET_POS_OFFSET(pos, dx, dy) ((void *)((((uint8_t *) (pos).ptr) + (dx) * (int64_t) (pos).elem_size + (dy) * (int64_t) (pos).row_length)))
    template <typename elem_t>
    elem_t *offset(int32_t dx, int32_t dy) {
      return (elem_t *) GET_POS_OFFSET(*this, dx, dy);
    }

    #define GET_ELEM(pos) ((pos).ptr)
    template <typename elem_t>
    elem_t *get() const {
      return (elem_t *) GET_ELEM(*this);
    }
  };

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

  using do_fun_t = void (Grid2DInfo::Pos);
  
  template<do_fun_t fun_ptr>
  void cellDo(uint32_t border = 0) {
    //TODO: openMP?
    for (unsigned y = border; y < info.height - border; y++)
      for (unsigned x = border; x < info.width - border; x++)
        fun_ptr(info.at(data.data(), x, y));
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

  using do_fun_t = Grid2DImpl::do_fun_t;
  using update_fun_t = elem_t (Grid2DInfo::Pos);
  
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
        *info.at(data_copy.data(), x, y).get<elem_t>() = fun_ptr(info.at(data.data(), x, y));
    
    data = std::move(data_copy);
  }
};

} // namespace cw

#endif // __GRID2D_H__