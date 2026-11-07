cmake_minimum_required(VERSION 3.20)

project(InspectFlowInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 图片预处理、热力图生成和 PNG 编码分别使用这些 OpenCV 模块。
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

set(ONNXRUNTIME_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-osx-arm64-1.30.0"
    CACHE PATH "ONNX Runtime SDK directory"
)

set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ORT_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")

if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime C++ header not found: ${ORT_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${ORT_LIBRARY}")
    message(FATAL_ERROR "ONNX Runtime library not found: ${ORT_LIBRARY}")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
)

add_library(inspectflow_core STATIC
    src/preprocess.cpp
    src/inference_engine.cpp
    src/result_io.cpp
)

target_include_directories(inspectflow_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(inspectflow_core PUBLIC
    onnxruntime
    ${OpenCV_LIBS}
)

add_executable(inspectflow_infer src/main.cpp)
target_link_libraries(inspectflow_infer PRIVATE inspectflow_core)

# Let the executable find the bundled ONNX Runtime dylib from the build tree.
set_target_properties(inspectflow_infer PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

include(CTest)

if(BUILD_TESTING)
    add_executable(test_preprocess tests/test_preprocess.cpp)
    target_link_libraries(test_preprocess PRIVATE inspectflow_core)
    add_test(NAME preprocess COMMAND test_preprocess)
endif()

find_package(Threads REQUIRED)

add_executable(inspectflow_server
    src/server_main.cpp
)

target_include_directories(inspectflow_server PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpp-httplib"
)

target_link_libraries(inspectflow_server PRIVATE
    inspectflow_core
    Threads::Threads
)

set_target_properties(inspectflow_server PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
