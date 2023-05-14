#include <iostream>
#include "CudaWrappers.h"
#include "BMPImage.h"

int main() {
  std::cout << "List of available devices: " << std::endl;
  for (auto& device : cw::getDevices()) {
    std::cout << device << std::endl;
  }

  cw::BMPImage bmp(2048, 555);
  for (unsigned y = 0; y < bmp.height; y++) {
    for (unsigned x = 0; x < bmp.width; x++) {
      auto &pixel = bmp.at(x, y);
      pixel.r = x * 255 / bmp.width;
      pixel.g = y * 255 / bmp.height;
    }
  }
  bmp.saveToFile("img.bmp");
}