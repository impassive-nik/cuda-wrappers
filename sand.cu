#include "BMPImage.h"
#include "CudaWrappers.h"
#include "Grid2D.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

enum class Px : char {
  EMPTY,
  SAND,
  WALL,
  END = WALL
};

inline cw::BMPImage::ConstPixel ToPixel(Px px, unsigned x, unsigned y) {
  switch (px) {
  case Px::EMPTY:
    return {255, 255, 255};
  case Px::SAND:
    return {255, 255, 0};
  case Px::WALL:
    return {255, 25, 25};
  }
}

__device__ __host__ void init(Px &px, unsigned x, unsigned y, unsigned w, unsigned h) {
  px = (x == 0 || y == 0 || y == h - 1 || x == w - 1) ? Px::WALL : Px::EMPTY;
}

__device__ __host__ void calculate(Px *px) {
  if (*px == Px::EMPTY)
    *px = Px::SAND;
  else if (*px == Px::SAND)
    *px = Px::EMPTY;
}

int main() {
  std::cout << "Step 0" << std::endl;
  cw::Grid2D<Px> grid(48, 48);
  cw::DeviceGrid2D<Px> grid_dev(grid);

  for (unsigned y = 0; y < grid.height; y++)
    for (unsigned x = 0; x < grid.width; x++)
      init(grid.at(x, y), x, y, grid.width, grid.height);

  std::cout << "Step 1" << std::endl;
  for (int i = 0; i < 10; i++) {
    for (unsigned y = 0; y < grid.height; y++)
      for (unsigned x = 0; x < grid.width; x++)
        calculate(&grid.at(x, y));

    saveToBMP(grid, ToPixel, "img_" + std::to_string(i) + ".bmp");
  }
  std::cout << "Step 2" << std::endl;
  grid_dev.copyToDevice();
  for (int i = 0; i < 10; i++) {
    grid_dev.calculate(calculate);

    grid_dev.copyToHost();
    saveToBMP(grid, ToPixel, "dev_img_" + std::to_string(i) + ".bmp");
  }
  std::cout << "End" << std::endl;
}