#include <opencv2/opencv.hpp>
#include "types.hpp"


// visual additions not part of the main concept
namespace testing::visuals
{
    // feedback on "detect_document" algorithm that requires correctly ordered
    auto annotate_document(const cv::Mat& image, const Parallelogram& points) -> void
    {
        // circle
        static constexpr auto circle_radius = 10;
        static const cv::Scalar circle_color {0, 0, 0xff};

        // lines
        static constexpr auto line_thickness = 2;
        static const auto& line_color = circle_color;

        for (const auto& point : points)
        {
            cv::circle(image, point, circle_radius, circle_color, cv::FILLED);
        }

        cv::polylines(image, points, true, line_color, line_thickness);
    }
}