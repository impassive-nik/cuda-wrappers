#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__

#include <chrono>
#include <string>

namespace cw {

class Stopwatch {
  std::chrono::time_point<std::chrono::high_resolution_clock> created =
      std::chrono::high_resolution_clock::now();
  std::chrono::time_point<std::chrono::high_resolution_clock> last;
public:
  Stopwatch() {
    reset();
  }

  void reset() {
    last = std::chrono::high_resolution_clock::now();    
  }

  std::string getLastTime() {
    return std::to_string(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - last).count()) + " seconds";
  }

  std::string getTotalTime() {
    return std::to_string(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - created).count()) + " seconds";
  }

};

}

#endif // __STOPWATCH_H__