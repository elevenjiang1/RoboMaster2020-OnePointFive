cmake_minimum_required(VERSION 3.0.0)
project(OpenVINO_TEST)
# find_package(ngraph REQUIRED)
# find_package(InferenceEngine REQUIRED)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


set(OpenCV_DIR "/opt/intel/openvino_2020.1.023/opencv/cmake")

set(Torch_DIR "/home/elevenjiang/Documents/ALL_code/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)


find_package(OpenCV 4.2 REQUIRED)
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# target_link_libraries(${PROJECT_NAME} PRIVATE ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} ${NGRAPH_LIBRARIES})