CMAKE_MINIMUM_REQUIRED(VERSION 3.6 FATAL_ERROR)

project(libfastpd)


find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter Development)

SET(LIBRARY_OUTPUT_PATH  ${PROJECT_BINARY_DIR}/..)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}       # 这会对应到 src2
    ${Python3_INCLUDE_DIRS}           # Python3 的 include 目录
)
link_directories(${Python3_LIBRARY_DIRS})

ADD_DEFINITIONS(-fPIC)
ADD_DEFINITIONS(--std=c++11)

add_definitions(-fPIC)
add_definitions(--std=c++11)

ADD_LIBRARY(fastpd
    SHARED
    fastpd/FastPD.cpp
    fastpd/FastPD.h
    fastpd/block.h
    fastpd/graph.h
    fastpd/graph.cpp
    init_fastpd.cpp
    )

target_link_libraries(fastpd PRIVATE ${Python3_LIBRARIES})
