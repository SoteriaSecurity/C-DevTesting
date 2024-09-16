cmake_minimum_required(VERSION 3.10)
project(C_DevTesting)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS log)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# Add executable
add_executable(YourProjectName main.cpp)

# Link libraries
target_link_libraries(YourProjectName ${OpenCV_LIBS} Boost::log)
