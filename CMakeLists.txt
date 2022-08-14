cmake_minimum_required(VERSION 3.5)
project(LFTools)

find_package(OpenCV REQUIRED)
include_directories(include)

file(GLOB SOURCES "src/*.cpp")
add_executable(LFTools ${SOURCES})
target_link_libraries(LFTools ${OpenCV_LIBS})
