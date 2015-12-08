cmake_minimum_required(VERSION 3.2)

project("benchmark-superpixels")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3 -msse4.2 ")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem program_options system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(HEADER_FILES src/segmentation.h src/utils.h src/seeds.h src/donald-slic.h src/coursetofine.h src/contrib/coursetofine/functions.h src/contrib/coursetofine/segengine.h src/contrib/coursetofine/stdafx.h src/contrib/coursetofine/structures.h src/contrib/coursetofine/targetver.h src/contrib/coursetofine/tsdeque.h src/contrib/coursetofine/utils.h)
set(SOURCE_FILES src/segmentation.cpp src/utils.cpp src/seeds.cpp src/donald-slic.cpp src/coursetofine.cpp src/contrib/coursetofine/stdafx.cpp src/contrib/coursetofine/utils.cpp src/contrib/coursetofine/functions.cpp src/contrib/coursetofine/structures.cpp src/contrib/coursetofine/segengine.cpp)

add_executable(benchmark-superpixels ${SOURCE_FILES} ${HEADER_FILES} src/benchmark-superpixels.cpp)

add_executable(run-slic ${SOURCE_FILES} ${HEADER_FILES} src/run-donald-slic.cpp )
add_executable(run-seeds ${SOURCE_FILES} ${HEADER_FILES} src/run-seeds.cpp)
add_executable(run-coursetofine ${SOURCE_FILES} ${HEADER_FILES} src/run-coursetofine.cpp)

target_link_libraries(benchmark-superpixels ${OpenCV_LIBS})
target_link_libraries(run-slic ${OpenCV_LIBS})
target_link_libraries(run-seeds ${OpenCV_LIBS})
target_link_libraries(run-coursetofine ${OpenCV_LIBS})
target_link_libraries(benchmark-superpixels ${Boost_LIBRARIES})