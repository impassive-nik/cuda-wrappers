#ifndef __GRID2D_H__
#define __GRID2D_H__

#include <stdint.h>
#include <vector>

namespace cw {

struct Grid2DImpl {
  const uint32_t width;
  const uint32_t height;

  const uint32_t row_padding;
  const uint32_t row_length;

  const size_t elem_size;

  std::vector<uint8_t> data;

  Grid2DImpl(size_t elem_size, uint32_t width, uint32_t height,
             uint32_t pad = 1)
      : elem_size(elem_size), width(width), height(height),
        row_padding((pad - ((elem_size * width) % pad)) % pad),
        row_length(elem_size * width + row_padding),
        data(row_length * height) {}

  const void *at(uint32_t x, uint32_t y) const {
    return &data[row_length * y + x * elem_size];
  }

  void *at(uint32_t x, uint32_t y) {
    return &data[row_length * y + x * elem_size];
  }
};

template <typename elem_t>
struct Grid2D : public Grid2DImpl {
  Grid2D(uint32_t width, uint32_t height, uint32_t pad = 1)
      : Grid2DImpl(sizeof(elem_t), width, height, pad) {}

  const elem_t &at(uint32_t x, uint32_t y) const {
    return *((const elem_t *) Grid2DImpl::at(x, y));
  }

  elem_t &at(uint32_t x, uint32_t y) {
    return *((elem_t *) Grid2DImpl::at(x, y));
  }
};

} // namespace cw

#endif // __GRID2D_H__