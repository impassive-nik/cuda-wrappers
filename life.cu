#include "BMPImage.h"
#include "CudaWrappers.h"
#include "Grid2D.h"
#include "DeviceGrid2D.h"
#include "Stopwatch.h"

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

enum class Cell : char {
  EMPTY = 0,
  CELL = 1
};

inline cw::BMPImage::ConstPixel toPixel(Cell c, unsigned x, unsigned y) {
  switch (c) {
  case Cell::EMPTY:
    return {255, 255, 255};
  case Cell::CELL:
    return {0, 255, 0};
  }
  return {0, 0, 0};
}

const size_t BORDER = 1;
const size_t WIDTH = 128;
const size_t HEIGHT = WIDTH;

void init(cw::Grid2DInfo::Pos pos) {
  auto &result = *((Cell *) GET_ELEM(pos));
  result = (rand() % 3 == 1) ? Cell::CELL : Cell::EMPTY;
}

CUDA_HOSTDEV Cell move(cw::Grid2DInfo::Pos pos) {
  auto c = *((char *) GET_ELEM(pos));

  int neighbours = 0;
  neighbours += *((char *) GET_POS_OFFSET(pos, -1, -1));
  neighbours += *((char *) GET_POS_OFFSET(pos,  0, -1));
  neighbours += *((char *) GET_POS_OFFSET(pos,  1, -1));

  neighbours += *((char *) GET_POS_OFFSET(pos,  -1, 0));
  neighbours += *((char *) GET_POS_OFFSET(pos,   1, 0));

  neighbours += *((char *) GET_POS_OFFSET(pos, -1,  1));
  neighbours += *((char *) GET_POS_OFFSET(pos,  0,  1));
  neighbours += *((char *) GET_POS_OFFSET(pos,  1,  1));

  if (c) {
    return (Cell)(neighbours > 1 && neighbours < 4);
  } else {
    return (Cell)(neighbours == 3);
  }
}

int main() {
  const int ITERS = 10;
  const bool RUN_CPU = true;
  const bool RUN_GPU = true;

  cw::Stopwatch timer;
  srand((unsigned) time(NULL));

  // ----------------------------------------------------------------
  std::cout << " --- Init --- " << std::endl;

  cw::Grid2D<Cell>       grid_cpu(WIDTH, HEIGHT);
  cw::DeviceGrid2D<Cell> grid_gpu(WIDTH, HEIGHT);

  grid_cpu.cellDo<init>(BORDER);
  grid_gpu.copyFrom(grid_cpu);

  std::cout << "Execution time: " + timer.getLastTime() << std::endl;

  saveToBMP(grid_cpu, toPixel, "life_0.bmp");

  timer.reset();

  // ----------------------------------------------------------------
  if (RUN_CPU) {
    std::cout << " --- Step 1 - CPU --- " << std::endl;

    for (int i = 0; i < ITERS; i++) {
      grid_cpu.cellUpdate<move>(BORDER);
      saveToBMP(grid_cpu, toPixel, "life_cpu_" + std::to_string(i + 1) + ".bmp");
    }

    std::cout << "Execution time: " + timer.getLastTime() << std::endl;

    timer.reset();
  }
  
  // ----------------------------------------------------------------
  if (RUN_GPU) {
    std::cout << " --- Step 2 - GPU --- " << std::endl;

    for (int i = 0; i < ITERS; i++) {
      grid_gpu.cellUpdate<move>(BORDER);
      grid_gpu.copyTo(grid_cpu);
      saveToBMP(grid_cpu, toPixel, "life_gpu_" + std::to_string(i + 1) + ".bmp");
    }

    std::cout << "Execution time: " + timer.getLastTime() << std::endl;
  
    timer.reset();
  }
  std::cout << " --- End --- " << std::endl;

  std::cout << "Total Execution time: " + timer.getTotalTime() << std::endl;
}