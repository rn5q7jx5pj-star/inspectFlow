cmake_minimum_required(VERSION 3.20)

project(InspectFlowInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 可通过 cmake -DONNXRUNTIME_ROOT=... 修改依赖位置。
set(ONNXRUNTIME_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-osx-arm64-1.30.0"
    CACHE PATH "ONNX Runtime SDK directory"
)

set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ORT_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")

if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime header not found: ${ORT_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${ORT_LIBRARY}")
    message(FATAL_ERROR "ONNX Runtime library not found: ${ORT_LIBRARY}")
endif()

# 告诉CMake：这是外部已经编译好的动态库。
add_library(onnxruntime SHARED IMPORTED)

set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
)

add_executable(inspectflow_infer src/main.cpp)

target_link_libraries(inspectflow_infer PRIVATE onnxruntime)

# 让构建目录里的程序运行时能够找到动态库。
set_target_properties(inspectflow_infer PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

# 查找本次需要的三个OpenCV模块。
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

# 将OpenCV接入现有推理程序。
target_include_directories(
    inspectflow_infer PRIVATE ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    inspectflow_infer PRIVATE ${OpenCV_LIBS}
)