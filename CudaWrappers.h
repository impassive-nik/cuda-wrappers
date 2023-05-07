#ifndef __CUDA_WRAPPERS_H__
#define __CUDA_WRAPPERS_H__

#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cw {

// returns memory size in normalized format, e.g. "512 Bytes", "2.5 Gb" etc.
std::string memsizeToString(size_t bytes);

struct DeviceInfo {
  int id = -1;
  std::string name;
  size_t mem_total = 0;
  size_t mem_shared_per_block = 0;
  size_t warp_size = 0;

  std::string to_string() const {
    std::stringstream ss;
    ss << "#" << id << " - " << name << std::endl;
    ss << "  Total global memory: " << memsizeToString(mem_total) << std::endl;
    ss << "  Shared memory per block: " << memsizeToString(mem_shared_per_block) << std::endl;
    ss << "  Warp-size: " << memsizeToString(warp_size) << std::endl;
    return ss.str();
  }
};

std::vector<DeviceInfo> getDevices();

std::ostream& operator<<(std::ostream& os, const DeviceInfo& di);

}



#endif // __CUDA_WRAPPERS_H__