#ifndef __BMP_IMAGE_H__
#define __BMP_IMAGE_H__

#include "Grid2D.h"

#include <array>
#include <iostream>
#include <vector>

namespace cw {

struct BMPImage : Grid2D<uint8_t[3]> {
  using raw_elem_t = uint8_t[3];

  BMPImage(uint32_t width, uint32_t height) : Grid2D(width, height, 4) {}

  struct ConstPixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;

    ConstPixel(const raw_elem_t& elem): r(elem[0]), g(elem[1]), b(elem[2]) {
    }
  };

  struct Pixel {
    uint8_t &r;
    uint8_t &g;
    uint8_t &b;

    Pixel(raw_elem_t &elem) : r(elem[0]), g(elem[1]), b(elem[2]) {}
  };

  ConstPixel at(uint32_t x, uint32_t y) const { 
    return ConstPixel(Grid2D::at(x, height - 1 - y));
  }

  Pixel at(uint32_t x, uint32_t y) {
    return Pixel(Grid2D::at(x, height - 1 - y));
  }

  void saveToFile(const std::string &filename);
};

} // namespace cw

std::ostream &operator<<(std::ostream &os, const cw::BMPImage &img);

#endif __BMP_IMAGE_H__