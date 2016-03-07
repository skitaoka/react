cmake_minimum_required(VERSION 2.8)
project(stream_and_event)

find_package(CUDA REQUIRED)

cuda_add_executable(stream_and_event src/main.cu)
