#include "BMPImage.h"
#include "CudaWrappers.h"
#include "Grid2D.h"
#include "DeviceGrid2D.h"
#include "Stopwatch.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

enum class Px : char {
  EMPTY,
  SAND,
  WALL,
  END = WALL
};

inline cw::BMPImage::ConstPixel toPixel(Px px, unsigned x, unsigned y) {
  switch (px) {
  case Px::EMPTY:
    return {255, 255, 255};
  case Px::SAND:
    return {155, 155, 0};
  case Px::WALL:
    return {255, 25, 25};
  }
  return {0, 0, 0};
}

const size_t BORDER = 10;
const size_t WIDTH = 4096;
const size_t HEIGHT = WIDTH;

CUDA_HOSTDEV void init(cw::Grid2DInfo::Pos pos) {
  auto x = GET_POS_X(pos);
  auto y = GET_POS_Y(pos);
  auto &result = *((Px *) GET_ELEM(pos));

  if (x < BORDER || y < BORDER || y >= HEIGHT - BORDER || x >= WIDTH - BORDER)
    result = Px::WALL;
  else if ((x * 3 + y) % 13 == 4)
    result = Px::SAND;
  else
    result = Px::EMPTY;
}

CUDA_HOSTDEV Px move(cw::Grid2DInfo::Pos pos) {
  auto px = (Px *) GET_ELEM(pos);
  auto px_above = (Px *) GET_POS_OFFSET(pos, 0, -1);
  auto px_below = (Px *) GET_POS_OFFSET(pos, 0, 1);

  if (*px == Px::EMPTY && *px_above == Px::SAND)
    return Px::SAND;
  if (*px == Px::SAND && *px_below == Px::EMPTY)
    return Px::EMPTY;

  return *px;
}

int main() {
  const int ITERS = 100;
  const bool RUN_CPU = true;
  const bool RUN_GPU = true;

  cw::Stopwatch timer;

  // ----------------------------------------------------------------
  std::cout << " --- Init --- " << std::endl;

  cw::Grid2D<Px>       grid_cpu(WIDTH, HEIGHT);
  cw::DeviceGrid2D<Px> grid_gpu(WIDTH, HEIGHT);

  if (RUN_CPU)
    grid_cpu.cellDo<init>();

  if (RUN_GPU)
    grid_gpu.cellDo<init>();

  std::cout << "Execution time: " + timer.getLastTime() << std::endl;

  saveToBMP(grid_cpu, toPixel, "sand_0.bmp");

  timer.reset();

  // ----------------------------------------------------------------
  if (RUN_CPU) {
    std::cout << " --- Step 1 - CPU --- " << std::endl;

    for (int i = 0; i < ITERS; i++)
      grid_cpu.cellUpdate<move>(BORDER);


    std::cout << "Execution time: " + timer.getLastTime() << std::endl;
              
    saveToBMP(grid_cpu, toPixel, "sand_cpu.bmp");
  
    timer.reset();
  }
  
  // ----------------------------------------------------------------
  if (RUN_GPU) {
    std::cout << " --- Step 2 - GPU --- " << std::endl;

    for (int i = 0; i < ITERS; i++)
      grid_gpu.cellUpdate<move>(BORDER);

    grid_gpu.copyTo(grid_cpu);

    std::cout << "Execution time: " + timer.getLastTime() << std::endl;

    saveToBMP(grid_cpu, toPixel, "sand_gpu.bmp");
  
    timer.reset();
  }
  std::cout << " --- End --- " << std::endl;

  std::cout << "Total Execution time: " + timer.getTotalTime() << std::endl;
}