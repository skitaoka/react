cmake_minimum_required(VERSION 2.8)
project(stream_callback)

find_package(CUDA REQUIRED)

cuda_add_executable(stream_callback src/main.cu)
