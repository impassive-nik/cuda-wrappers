#ifndef __BMP_IMAGE_H__
#define __BMP_IMAGE_H__

#include <array>
#include <iostream>
#include <vector>

namespace cw {


struct BMPImage {
  const uint32_t width;
  const uint32_t height;

  const uint32_t padding;
  const uint32_t row_length;

  std::vector<uint8_t> data;

  struct Pixel {
    uint8_t &r;
    uint8_t &g;
    uint8_t &b;
    Pixel(uint8_t *p) : r(*p), g(p[1]), b(p[2]) {}
  };

  BMPImage(uint32_t width, uint32_t height)
      : width(width), height(height), 
        padding((4 - ((3 * width) % 4)) % 4), 
        row_length(3 * width + padding),
        data(row_length * height) {
  }

  Pixel at(uint32_t x, uint32_t y) {
    return Pixel(&data[row_length * (height - 1 - y) + x * 3]);
  }

  void saveToFile(const std::string &filename);
};

} // namespace cw

std::ostream &operator<<(std::ostream &os, const cw::BMPImage &img);

#endif __BMP_IMAGE_H__