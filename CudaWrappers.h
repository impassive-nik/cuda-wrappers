#ifndef __CUDA_WRAPPERS_H__
#define __CUDA_WRAPPERS_H__

#include <vector>
#include <string>

namespace cw {

struct DeviceInfo {
  int id;
  std::string name;
};


std::vector<DeviceInfo> getDevices();

}



#endif // __CUDA_WRAPPERS_H__