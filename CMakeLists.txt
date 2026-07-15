cmake_minimum_required(VERSION 3.14)
project(ezafe_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ONNX Runtime (same as hush_cpp)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)
find_path(ONNX_RUNTIME_INCLUDE onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime)

if(NOT ONNX_RUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found")
endif()
if(NOT ONNX_RUNTIME_INCLUDE)
    message(FATAL_ERROR "ONNX Runtime headers not found")
endif()

message(STATUS "ONNX Runtime lib: ${ONNX_RUNTIME_LIB}")
message(STATUS "ONNX Runtime include: ${ONNX_RUNTIME_INCLUDE}")

set(SOURCES
    src/ezafe_detector.cpp
    src/main.cpp
    third_party/sentencepiece_minimal/sentencepiece_minimal.cpp
)

add_executable(ezafe_detector ${SOURCES})
target_include_directories(ezafe_detector PRIVATE
    src
    third_party/sentencepiece_minimal
    ${ONNX_RUNTIME_INCLUDE}
)
target_link_libraries(ezafe_detector PRIVATE
    ${ONNX_RUNTIME_LIB}
)

# Copy model directory to build output (same pattern as hush_cpp)
add_custom_command(TARGET ezafe_detector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/model
        $<TARGET_FILE_DIR:ezafe_detector>/model
    COMMENT "Copying ezafe model directory"
)