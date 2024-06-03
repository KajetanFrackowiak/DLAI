cmake_minimum_required(VERSION 3.0)
project(WritePNG)

add_executable(writepng writepng.c)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
target_link_libraries(writepng ${PNG_LIBRARY})

set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR/bin})
