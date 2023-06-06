#include <iostream>
#include "CudaWrappers.h"
#include "Grid2D.h"
#include "BMPImage.h"

struct Cell {
  int x = 0;
  int y = 0;

  Cell() {}
  Cell(int x, int y): x(x), y(y) {
  }

  static cw::BMPImage::ConstPixel toPixel(const Cell &cell, unsigned x, unsigned y) {
    return {(uint8_t)(cell.x % 256),
            (uint8_t)(cell.y % 256),
            (uint8_t)((cell.x + cell.y) % 256)};
  }
};

int main() {
  std::cout << "List of available devices: " << std::endl;
  for (auto& device : cw::getDevices())
    std::cout << device << std::endl;

  cw::Grid2D<Cell> grid(2048, 555);
  for (unsigned y = 0; y < grid.info.height; y++)
    for (unsigned x = 0; x < grid.info.width; x++)
      *grid.at(x, y) = Cell(x, y);

  cw::BMPImage bmp(grid, Cell::toPixel);
  bmp.saveToFile("img.bmp");
}