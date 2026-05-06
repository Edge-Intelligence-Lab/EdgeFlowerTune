cmake_minimum_required(VERSION 3.16)
project(lshaped_flower_client LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LSHAPED_ENABLE_MFT "Enable MobileFineTuner-backed prefix encoder" ON)
option(LSHAPED_REGENERATE_PROTO "Regenerate Flower transport C++ from local protoc/grpc_cpp_plugin" OFF)
option(LSHAPED_ENABLE_BUNDLED_GRPC "Build gRPC/protobuf from bundled source when system packages are unavailable" OFF)
set(LSHAPED_BUNDLED_GRPC_SOURCE_DIR "" CACHE PATH "Pre-populated gRPC source tree for bundled Android builds")
set(LSHAPED_BUNDLED_GRPC_GIT_TAG "v1.68.0" CACHE STRING "gRPC tag used when fetching bundled Android builds")
set(
    LSHAPED_MFT_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/mobilefinetuner/operators"
    CACHE PATH
    "MobileFineTuner operators source tree"
)

if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    set(CMAKE_BUILD_RPATH
        "/usr/lib/x86_64-linux-gnu;/lib/x86_64-linux-gnu;$ENV{CONDA_PREFIX}/lib"
    )
endif()

find_package(gRPC CONFIG QUIET)
find_package(PkgConfig QUIET)
find_package(Threads REQUIRED)

set(LSHAPED_PROTOBUF_INCLUDE_DIR "")
set(LSHAPED_GRPC_LINK_LIBS "")
set(LSHAPED_PROTOBUF_LINK_LIBS "")
set(LSHAPED_PROTO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")
set(LSHAPED_PROTO_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/generated/flwr/proto/transport.pb.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/generated/flwr/proto/transport.grpc.pb.cc"
)

if(gRPC_FOUND AND NOT LSHAPED_ENABLE_BUNDLED_GRPC)
    find_path(PROTOBUF_INCLUDE_DIR
        NAMES google/protobuf/message.h
        HINTS "$ENV{CONDA_PREFIX}"
        PATH_SUFFIXES include
        REQUIRED
    )
    find_library(PROTOBUF_LIBRARY
        NAMES protobuf libprotobuf
        HINTS "$ENV{CONDA_PREFIX}"
        PATH_SUFFIXES lib
        REQUIRED
    )
    set(LSHAPED_PROTOBUF_INCLUDE_DIR "${PROTOBUF_INCLUDE_DIR}")
    set(LSHAPED_GRPC_LINK_LIBS gRPC::grpc++)
    set(LSHAPED_PROTOBUF_LINK_LIBS "${PROTOBUF_LIBRARY}")
elseif(LSHAPED_ENABLE_BUNDLED_GRPC)
    include(FetchContent)
    set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(gRPC_INSTALL OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_GRPC_CPP_PLUGIN OFF CACHE BOOL "" FORCE)
    set(gRPC_BUILD_CODEGEN OFF CACHE BOOL "" FORCE)
    set(gRPC_PROTOBUF_PROVIDER "module" CACHE STRING "" FORCE)
    set(gRPC_SSL_PROVIDER "module" CACHE STRING "" FORCE)
    set(gRPC_ZLIB_PROVIDER "module" CACHE STRING "" FORCE)
    set(gRPC_CARES_PROVIDER "module" CACHE STRING "" FORCE)
    set(gRPC_RE2_PROVIDER "module" CACHE STRING "" FORCE)
    set(gRPC_ABSL_PROVIDER "module" CACHE STRING "" FORCE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        set(CMAKE_SKIP_INSTALL_RULES ON CACHE BOOL "" FORCE)
    endif()
    set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "" FORCE)
    set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
    if(LSHAPED_BUNDLED_GRPC_SOURCE_DIR)
        get_filename_component(LSHAPED_BUNDLED_GRPC_PARENT_DIR "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}" DIRECTORY)
        if(EXISTS "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/abseil-cpp/CMakeLists.txt")
            set(ABSL_ROOT_DIR "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/abseil-cpp" CACHE PATH "" FORCE)
        elseif(EXISTS "${LSHAPED_BUNDLED_GRPC_PARENT_DIR}/abseil-cpp/CMakeLists.txt")
            set(ABSL_ROOT_DIR "${LSHAPED_BUNDLED_GRPC_PARENT_DIR}/abseil-cpp" CACHE PATH "" FORCE)
        endif()
        if(EXISTS "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/protobuf/CMakeLists.txt")
            set(PROTOBUF_ROOT_DIR "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/protobuf" CACHE PATH "" FORCE)
        elseif(EXISTS "${LSHAPED_BUNDLED_GRPC_PARENT_DIR}/protobuf/CMakeLists.txt")
            set(PROTOBUF_ROOT_DIR "${LSHAPED_BUNDLED_GRPC_PARENT_DIR}/protobuf" CACHE PATH "" FORCE)
        endif()
        if(EXISTS "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/re2/CMakeLists.txt")
            set(RE2_ROOT_DIR "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/re2" CACHE PATH "" FORCE)
        elseif(EXISTS "${LSHAPED_BUNDLED_GRPC_PARENT_DIR}/re2/CMakeLists.txt")
            set(RE2_ROOT_DIR "${LSHAPED_BUNDLED_GRPC_PARENT_DIR}/re2" CACHE PATH "" FORCE)
        endif()
        if(EXISTS "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/zlib/CMakeLists.txt")
            set(ZLIB_ROOT_DIR "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/zlib" CACHE PATH "" FORCE)
        elseif(EXISTS "${LSHAPED_BUNDLED_GRPC_PARENT_DIR}/zlib/CMakeLists.txt")
            set(ZLIB_ROOT_DIR "${LSHAPED_BUNDLED_GRPC_PARENT_DIR}/zlib" CACHE PATH "" FORCE)
        endif()
        if(EXISTS "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/cares/cares/CMakeLists.txt")
            set(CARES_ROOT_DIR "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/cares/cares" CACHE PATH "" FORCE)
        endif()
        if(EXISTS "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/boringssl-with-bazel/CMakeLists.txt")
            set(BORINGSSL_ROOT_DIR "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}/third_party/boringssl-with-bazel" CACHE PATH "" FORCE)
        endif()
    endif()
    if(LSHAPED_BUNDLED_GRPC_SOURCE_DIR)
        FetchContent_Declare(
            grpc
            SOURCE_DIR "${LSHAPED_BUNDLED_GRPC_SOURCE_DIR}"
        )
    else()
        FetchContent_Declare(
            grpc
            URL "https://github.com/grpc/grpc/archive/refs/tags/${LSHAPED_BUNDLED_GRPC_GIT_TAG}.tar.gz"
            DOWNLOAD_EXTRACT_TIMESTAMP FALSE
        )
    endif()
    FetchContent_MakeAvailable(grpc)

    if(TARGET gRPC::grpc++)
        set(LSHAPED_GRPC_LINK_LIBS gRPC::grpc++)
    elseif(TARGET grpc++)
        set(LSHAPED_GRPC_LINK_LIBS grpc++)
    else()
        message(FATAL_ERROR "Bundled gRPC build completed but grpc++ target is missing")
    endif()

    if(TARGET protobuf::libprotobuf)
        set(LSHAPED_PROTOBUF_LINK_LIBS protobuf::libprotobuf)
    elseif(TARGET libprotobuf)
        set(LSHAPED_PROTOBUF_LINK_LIBS libprotobuf)
    else()
        message(FATAL_ERROR "Bundled gRPC build completed but protobuf target is missing")
    endif()
elseif(PkgConfig_FOUND AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
    pkg_check_modules(PROTOBUF_PKG REQUIRED IMPORTED_TARGET protobuf)
    set(LSHAPED_PROTOBUF_INCLUDE_DIR "${PROTOBUF_PKG_INCLUDE_DIRS}")
    set(LSHAPED_GRPC_LINK_LIBS PkgConfig::GRPCPP)
    set(LSHAPED_PROTOBUF_LINK_LIBS PkgConfig::PROTOBUF_PKG)
else()
    message(FATAL_ERROR "Could not locate gRPC/protobuf toolchain")
endif()

if(LSHAPED_REGENERATE_PROTO)
    find_program(LSHAPED_PROTOC_EXECUTABLE protoc REQUIRED)
    find_program(LSHAPED_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin REQUIRED)
    set(LSHAPED_PROTO_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated")
    set(LSHAPED_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/generated/flwr/proto/transport.pb.cc"
        "${CMAKE_BINARY_DIR}/generated/flwr/proto/transport.grpc.pb.cc"
    )
    add_custom_command(
        OUTPUT
            "${CMAKE_BINARY_DIR}/generated/flwr/proto/transport.pb.cc"
            "${CMAKE_BINARY_DIR}/generated/flwr/proto/transport.pb.h"
            "${CMAKE_BINARY_DIR}/generated/flwr/proto/transport.grpc.pb.cc"
            "${CMAKE_BINARY_DIR}/generated/flwr/proto/transport.grpc.pb.h"
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated/flwr
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated/flwr/proto
        COMMAND ${LSHAPED_PROTOC_EXECUTABLE}
            -I ${CMAKE_CURRENT_SOURCE_DIR}/proto
            --cpp_out=${CMAKE_BINARY_DIR}/generated
            --grpc_out=${CMAKE_BINARY_DIR}/generated
            --plugin=protoc-gen-grpc=${LSHAPED_GRPC_CPP_PLUGIN_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/proto/flwr/proto/transport.proto
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/proto/flwr/proto/transport.proto
        VERBATIM
    )
endif()

add_library(lshaped_flower_proto STATIC
    ${LSHAPED_PROTO_SRCS}
)
target_include_directories(lshaped_flower_proto PUBLIC
    ${LSHAPED_PROTO_INCLUDE_DIR}
    ${LSHAPED_PROTOBUF_INCLUDE_DIR}
)
target_link_libraries(lshaped_flower_proto PUBLIC
    ${LSHAPED_PROTOBUF_LINK_LIBS}
    ${LSHAPED_GRPC_LINK_LIBS}
)

set(LSHAPED_CLIENT_SOURCES
    src/client_config.cpp
    src/federated_trainer.cpp
    src/prefix_encoder.cpp
    src/mmlu_dataset.cpp
    src/npy_serializer.cpp
    src/flower_legacy_client.cpp
    src/split_flower_client.cpp
    src/runner.cpp
)

add_library(lshaped_client_core STATIC ${LSHAPED_CLIENT_SOURCES})
target_include_directories(lshaped_client_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LSHAPED_PROTO_INCLUDE_DIR}
)
target_link_libraries(lshaped_client_core PUBLIC
    lshaped_flower_proto
    Threads::Threads
)
target_compile_definitions(lshaped_client_core PUBLIC
    $<$<BOOL:${LSHAPED_ENABLE_MFT}>:LSHAPED_ENABLE_MFT>
)

if(LSHAPED_ENABLE_MFT)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        set(USE_BLAS OFF CACHE BOOL "" FORCE)
    endif()
    set(USE_HF_TOKENIZERS ON CACHE BOOL "" FORCE)
    if(NOT EXISTS "${LSHAPED_MFT_SOURCE_DIR}/finetune_ops/core/tokenizer_hf.cpp")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../operators/finetune_ops/core/tokenizer_hf.cpp")
            set(LSHAPED_MFT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../operators")
        else()
            message(FATAL_ERROR "Could not locate MobileFineTuner operators source tree")
        endif()
    endif()
    add_subdirectory(
        ${LSHAPED_MFT_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/mobilefinetuner_operators
    )
    target_include_directories(lshaped_client_core PUBLIC
        ${LSHAPED_MFT_SOURCE_DIR}
    )
    target_link_libraries(lshaped_client_core PUBLIC operators)
endif()

add_executable(lshaped_flower_client src/main.cpp)
target_link_libraries(lshaped_flower_client PRIVATE lshaped_client_core)
