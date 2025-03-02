cmake_minimum_required(VERSION 3.30)
project(ppp_ultimate_cuda CUDA)

set(CMAKE_CUDA_STANDARD 26)

add_executable(ppp_ultimate_cuda main.cu
        Commons.cu
        Commons.hpp
        Render.cu
        Render.hpp
        Sphere.hpp
        Vec.hpp
)

set_target_properties(ppp_ultimate_cuda PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

add_compile_options(-g -G)
