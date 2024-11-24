cmake_minimum_required(VERSION 3.11.3)
project(2d_feature_matching)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(rerun_sdk REQUIRED)

file(GLOB SOURCES src/*.cpp)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${rerun_sdk_INCLUDE_DIRS}
)

link_libraries(${OpenCV_LIBRARIES} rerun_sdk)

file(GLOB_RECURSE src_files src/*.cpp)

add_executable (main ${src_files})
