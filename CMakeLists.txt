cmake_minimum_required(VERSION 3.2)

project("benchmark-superpixels")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Ofast -march=native")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem program_options system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(HEADER_FILES src/segmentation.h src/utils.h src/seeds.h src/donald-slic.h src/coarsetofine.h src/contrib/coarsetofine/functions.h src/contrib/coarsetofine/segengine.h src/contrib/coarsetofine/stdafx.h src/contrib/coarsetofine/structures.h src/contrib/coarsetofine/targetver.h src/contrib/coarsetofine/tsdeque.h src/contrib/coarsetofine/utils.h)
set(SOURCE_FILES src/segmentation.cpp src/utils.cpp src/seeds.cpp src/donald-slic.cpp src/coarsetofine.cpp src/contrib/coarsetofine/stdafx.cpp src/contrib/coarsetofine/utils.cpp src/contrib/coarsetofine/functions.cpp src/contrib/coarsetofine/structures.cpp src/contrib/coarsetofine/segengine.cpp)

add_executable(benchmark-single ${SOURCE_FILES} ${HEADER_FILES} src/benchmark-single.cpp)

add_executable(run-slic ${SOURCE_FILES} ${HEADER_FILES} src/run-donald-slic.cpp )
add_executable(run-seeds ${SOURCE_FILES} ${HEADER_FILES} src/run-seeds.cpp)
add_executable(run-coarsetofine ${SOURCE_FILES} ${HEADER_FILES} src/run-coarsetofine.cpp)


target_link_libraries(benchmark-single ${OpenCV_LIBS})
target_link_libraries(run-slic ${OpenCV_LIBS})
target_link_libraries(run-seeds ${OpenCV_LIBS})
target_link_libraries(run-coarsetofine ${OpenCV_LIBS})

target_link_libraries(benchmark-single ${Boost_LIBRARIES})
