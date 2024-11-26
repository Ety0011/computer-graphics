cmake_minimum_required(VERSION 3.28)
project(computer-graphics VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

set(PROFILES vanilla consts_only bvh_moller)

foreach(PROFILE ${PROFILES})
    file(GLOB SRCS "Bonus1/code/*.cpp" "Bonus1/code/${PROFILE}/*.cpp")
    add_executable(${PROFILE} ${SRCS})
    target_compile_definitions(${PROFILE} PRIVATE PROFILE_NAME=${PROFILE})
    target_include_directories(${PROFILE} PRIVATE .)
endforeach()
