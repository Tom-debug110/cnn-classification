cmake_minimum_required(VERSION 3.23)
project(cnn)

set(CMAKE_CXX_STANDARD 20)


find_package(OpenCV)

include_directories(include)
add_subdirectory(src )
