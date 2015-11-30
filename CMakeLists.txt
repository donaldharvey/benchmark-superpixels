cmake_minimum_required(VERSION 3.2)

project("benchmark-superpixels")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -O3 -msse4.2")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem program_options system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(HEADER_FILES src/segmentation.h src/seeds.h src/donald-slic.h)
set(SOURCE_FILES src/segmentation.cpp src/seeds.cpp src/donald-slic.cpp src/benchmark-superpixels.cpp)

add_executable(benchmark-superpixels ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(benchmark-superpixels ${OpenCV_LIBS})
target_link_libraries(benchmark-superpixels ${Boost_LIBRARIES})