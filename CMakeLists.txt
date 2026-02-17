cmake_minimum_required(VERSION 3.18)
project(noneq_statmech LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# CUDA 13.1 doesn't officially support VS 2026 yet - override the check
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")

include_directories(${PROJECT_SOURCE_DIR}/include)

# --- Linear algebra library (shared by everything) ---
add_library(linalg src/linalg/linalg.cu)
target_link_libraries(linalg cusolver cublas)

# --- Main simulation executable ---
add_executable(quench_sim
    src/main.cu
    src/quenches/quenches.cu
)
target_link_libraries(quench_sim linalg)

# --- Tests ---
add_executable(test_diagonalize tests/test_diagonalize.cu)
target_link_libraries(test_diagonalize linalg)
