project(ncnnyolox)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc highgui)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230223-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

include_directories (${INCLUDE_DIRECTORIES} "${CMAKE_SOURCE_DIR}/MIPP/")

add_library(ncnnyolox SHARED yoloxncnn.cpp yolox.cpp ndkcamera.cpp line2Dup.cpp)

target_link_libraries(ncnnyolox ncnn ${OpenCV_LIBS} camera2ndk mediandk)
