cmake_minimum_required(VERSION 3.8)
project(test_project LANGUAGES CXX CUDA)

add_executable(main main.cpp CudaWrappers.cu BMPImage.cpp)
add_executable(sand sand.cu CudaWrappers.cu BMPImage.cpp)