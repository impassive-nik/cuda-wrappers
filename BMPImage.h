#ifndef __BMP_IMAGE_H__
#define __BMP_IMAGE_H__

#include "Grid2D.h"

#include <array>
#include <functional>
#include <iostream>
#include <vector>

namespace cw {

struct BMPImage : Grid2D<std::array<uint8_t, 3>> {
  using raw_elem_t = std::array<uint8_t, 3>;

  struct ConstPixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;

    ConstPixel(const raw_elem_t &elem) : r(elem[2]), g(elem[1]), b(elem[0]) {}
    ConstPixel(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}
  };

  struct Pixel {
    uint8_t &r;
    uint8_t &g;
    uint8_t &b;

    Pixel(raw_elem_t &elem) : r(elem[2]), g(elem[1]), b(elem[0]) {}

    Pixel& operator=(const Pixel& p) { 
      r = p.r;
      g = p.g;
      b = p.b;
      return *this;
    }

    Pixel &operator=(const ConstPixel &p) {
      r = p.r;
      g = p.g;
      b = p.b;
      return *this;
    }
  };

  BMPImage(uint32_t width, uint32_t height) : Grid2D(width, height, 4) {}

  template <typename elem_t, typename F>
  BMPImage(const Grid2D<elem_t> &grid, F f) : BMPImage(grid.info.width, grid.info.height) {
    for (uint32_t y = 0; y < info.height; y++)
      for (uint32_t x = 0; x < info.width; x++)
        at(x, y) = f(*grid.at(x, y), x, y);
  }

  ConstPixel at(uint32_t x, uint32_t y) const { 
    return ConstPixel(*Grid2D::at(x, info.height - 1 - y));
  }

  Pixel at(uint32_t x, uint32_t y) {
    return Pixel(*Grid2D::at(x, info.height - 1 - y));
  }

  void saveToFile(const std::string &filename);
};

std::ostream &operator<<(std::ostream &os, const cw::BMPImage &img);

template<typename elem_t, typename TO_PIXEL>
void saveToBMP(const Grid2D<elem_t> &grid, TO_PIXEL toPixel, const std::string &filename) {
  BMPImage bmp(grid, toPixel);
  bmp.saveToFile(filename);
}

} // namespace cw


#endif __BMP_IMAGE_H__