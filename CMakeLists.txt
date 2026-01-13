cmake_minimum_required(VERSION 3.24)
project(Lab7 C)
set(CMAKE_C_STANDARD 11)

add_compile_options(-Wall -march=native -O3)

add_executable(Lab7 main.c naive_edit_distance.c tiled_edit_distance.c parallelized_edit_distance.c avx2_edit_distance.c)
include_directories(src)

target_link_libraries(Lab7 pthread)

