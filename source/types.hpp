#pragma once

#include <array>
#include <string>
#include <opencv2/opencv.hpp>


// project for document detection and warping
namespace testing
{
    // TODO: extend "types.hpp" functionality


    // four points to identify complete position of a document in an image
    using Parallelogram = std::array<cv::Point, 4>;
}