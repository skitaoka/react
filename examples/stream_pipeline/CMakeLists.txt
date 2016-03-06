cmake_minimum_required(VERSION 2.8)
project(stream_pipeline)

find_package(CUDA REQUIRED)

cuda_add_executable(stream_pipeline src/main.cu)

