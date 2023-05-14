#ifndef __GRID2D_H__
#define __GRID2D_H__

#include <stdint.h>
#include <vector>

template<typename elem_t>
struct Grid2D {
  const uint32_t width;
  const uint32_t height;

  const uint32_t row_padding;
  const uint32_t row_length;

  std::vector<uint8_t> data;

  Grid2D(uint32_t width, uint32_t height, uint32_t pad = 1)
      : width(width), height(height),
        row_padding((pad - ((sizeof(elem_t) * width) % pad)) % pad),
        row_length(sizeof(elem_t) * width + row_padding),
        data(row_length * height) {
  }

  const elem_t &at(uint32_t x, uint32_t y) const {
    return *((const elem_t *) &data[row_length * y + x * sizeof(elem_t)]);
  }

  elem_t &at(uint32_t x, uint32_t y) {
    return *((elem_t *) &data[row_length * y + x * sizeof(elem_t)]);
  }
};

#endif // __GRID2D_H__