cmake_minimum_required(VERSION 3.16)
project(gemma_3_1b_pt_lora_finetune_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GEMMA_USE_HF_TOKENIZERS "Enable HF tokenizers-cpp (Rust) for Gemma" ON)
if(GEMMA_USE_HF_TOKENIZERS)
    set(USE_HF_TOKENIZERS ON CACHE BOOL "Enable HF tokenizers-cpp (from Gemma build)" FORCE)
endif()

add_subdirectory(../operators ${CMAKE_BINARY_DIR}/operators)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../operators)

add_executable(train
    src/main.cpp
)
target_link_libraries(train PRIVATE operators)

add_executable(eval_mmlu_gemma
    src/eval_mmlu_gemma.cpp
)
target_link_libraries(eval_mmlu_gemma PRIVATE operators)
