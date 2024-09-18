cmake_minimum_required(VERSION 3.10)
project(C_DevTesting)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch x86_64")
set(ONNXRUNTIME_DIR "/Users/omkarbantanur/Downloads/onnxruntime")





# Find packages
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS log log_setup thread filesystem system)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Add executable
add_executable(C_DevTesting main.cpp)

# Link libraries
target_link_libraries(C_DevTesting PRIVATE ${OpenCV_LIBS})
target_link_libraries(C_DevTesting PRIVATE Boost::log Boost::log_setup Boost::thread Boost::filesystem Boost::system)
target_link_libraries(C_DevTesting PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib)

