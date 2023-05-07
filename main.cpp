#include <iostream>
#include "CudaWrappers.h"

int main() {
  std::cout << "List of available devices: " << std::endl;
  for (auto& device : cw::getDevices()) {
    std::cout << device.id << " - " << device.name << std::endl;
  }
}