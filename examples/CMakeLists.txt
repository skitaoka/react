cmake_minimum_required(VERSION 2.8)
project(examples)

set(PLATFORM X64)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(stream_and_event)
add_subdirectory(stream_callback)
add_subdirectory(stream_pipeline)
