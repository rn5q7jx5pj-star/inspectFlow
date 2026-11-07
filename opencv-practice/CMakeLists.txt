cmake_minimum_required(VERSION 3.20)
project(OpenCVPractice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(opencv_practice main.cpp)

target_include_directories(
    opencv_practice PRIVATE ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    opencv_practice PRIVATE ${OpenCV_LIBS}
)