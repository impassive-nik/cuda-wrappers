#include "BMPImage.h"
#include <algorithm>
#include <fstream>

std::ostream &writeLE16(std::ostream &os, uint16_t value) {
  os.put((uint8_t) value);
  os.put((uint8_t) (value >> 8));
  return os;
}

std::ostream &writeLE32(std::ostream &os, uint32_t value) {
  os.put((uint8_t) value);
  os.put((uint8_t) (value >> 8));
  os.put((uint8_t) (value >> 16));
  os.put((uint8_t) (value >> 24));
  return os;
}

std::ostream &fillN(std::ostream &os, size_t n, uint8_t value) {
  std::fill_n(std::ostream_iterator<char>(os), n, value);
  return os;
}

std::ostream &operator<<(std::ostream &os, const cw::BMPImage &img) {
  uint32_t file_header_size = 14;
  uint32_t info_header_size = 40;
  uint32_t total_header_size = 40;
  uint32_t file_size = total_header_size + (uint32_t) img.data.size();

  // File header
  os << "BM";
  writeLE32(os, file_size);
  writeLE16(os, 0);
  writeLE16(os, 0);
  writeLE32(os, total_header_size);

  // Info header
  writeLE32(os, info_header_size);
  writeLE32(os, img.width);
  writeLE32(os, img.height);
  writeLE16(os, 1);  // Number of planes
  writeLE16(os, 24); // Bits per pixel
  fillN(os, info_header_size - 16 /* already written bytes count */, '\0');

  // Data
  os.write((const char *) &img.data[0], img.data.size());
  return os;
}

void cw::BMPImage::saveToFile(const std::string& filename) {
  std::ofstream outfile(filename, std::ios::out | std::ios::binary);
  outfile << *this;
  outfile.close();
}
