cmake_minimum_required(VERSION 3.16)
project(objectmodel_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(objectmodel_core SHARED twin_core.cpp)
target_link_libraries(objectmodel_core PRIVATE Eigen3::Eigen)
target_compile_options(objectmodel_core PRIVATE -Wall -Wextra -Wpedantic -Werror -O3)
set_target_properties(objectmodel_core PROPERTIES OUTPUT_NAME objectmodel_core)
