cmake_minimum_required(VERSION 2.8)
project(examples)

if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(PLATFORM1 32)
        set(PLATFORM2)
    else()
        set(PLATFORM1 64)
        set(PLATFORM2 _64)
    endif()

    if(PLATFORM1 EQUAL 64)
        add_definitions(-Dx64)
    endif()
endif(WIN32)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(stream_and_event)
add_subdirectory(stream_callback)
add_subdirectory(stream_pipeline)
