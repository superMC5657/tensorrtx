cmake_minimum_required(VERSION 3.14)

project(yolov5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/lib)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
#
#set(Cuda_DIR /home/supermc/envs/cuda-10.2-cudnn-7.6.5)
set(Cuda_DIR /usr/local/cuda)

include_directories(${Cuda_DIR}/include)
link_directories(${Cuda_DIR}/lib64)

# tensorrt
#set(TensorRT_DIR /home/supermc/envs/TensorRT/TensorRT-7.2.1.6)
set(TensorRT_DIR /home/supermc/envs/TensorRT/TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0/TensorRT-7.1.3.4)
include_directories(${TensorRT_DIR}/include)
link_directories(${TensorRT_DIR}/lib)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED src/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


set(Exe yolov5t)
add_executable(${Exe} src/yolov5.cpp)
target_link_libraries(${Exe} nvinfer)
target_link_libraries(${Exe} cudart)
target_link_libraries(${Exe} myplugins)
target_link_libraries(${Exe} ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

