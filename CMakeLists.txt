cmake_minimum_required(VERSION 3.22.1)
project(OpenPtxas)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(third_party/ELFIO)
add_subdirectory(sassasm)
add_subdirectory(sasslib)