cmake_minimum_required(VERSION 3.8)
project(test_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(main main.cpp CudaWrappers.cu BMPImage.cpp)
install(TARGETS main DESTINATION .)

add_executable(sand sand.cu CudaWrappers.cu BMPImage.cpp)
install(TARGETS sand DESTINATION .)

add_executable(life life.cu CudaWrappers.cu BMPImage.cpp)
install(TARGETS life DESTINATION .)