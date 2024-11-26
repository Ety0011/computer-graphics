cmake_minimum_required(VERSION 3.31)
project(computer-graphics VERSION 0.1.0 LANGUAGES C CXX)

add_executable(computer-graphics "Bonus 3/code/main.cpp")

find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    target_link_libraries(computer-graphics PUBLIC OpenMP::OpenMP_CXX)
endif()