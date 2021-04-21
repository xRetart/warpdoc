cmake_minimum_required(VERSION 3.13.4)

add_executable(warpdoc source/main.cpp)

include_directories(warpdoc PRIVATE /usr/local/include/opencv4/ dependencies/)

target_link_libraries(warpdoc PRIVATE libopencv_core.so libopencv_imgproc.so libopencv_videoio.so libopencv_highgui.so)
target_compile_features(warpdoc PRIVATE cxx_std_20)