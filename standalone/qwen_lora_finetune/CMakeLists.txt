cmake_minimum_required(VERSION 3.16)
project(qwen_lora_finetune_cli CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QWEN_USE_HF_TOKENIZERS "Enable native HF tokenizers-cpp (Rust) for Qwen" ON)
if(QWEN_USE_HF_TOKENIZERS)
    set(USE_HF_TOKENIZERS ON CACHE BOOL "Enable HF tokenizers-cpp (from Qwen build)" FORCE)
endif()

add_subdirectory(../operators ${CMAKE_BINARY_DIR}/operators)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../operators)

# Unified training entrypoint
add_executable(train
    src/main.cpp
)
target_link_libraries(train PRIVATE operators)

if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(train PRIVATE ${ACCELERATE_LIBRARY})
    endif()
endif()
