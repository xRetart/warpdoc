#include <array>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "aug/outparameter.hpp"
#include "types.hpp"


// algorithms for document detection and warping in order of optimal execution
namespace testing::processing
{
    using Dimensions = std::pair<int, int>;
    using FDimensions = std::pair<float, float>;


    auto reorder_points(const Parallelogram& points) -> Parallelogram;

    // math functions needed in processing
    namespace utility_math
    {
        [[nodiscard]] constexpr auto square(const auto value) noexcept
        {
            return value * value;
        }
        [[nodiscard]] auto distance(const std::pair<cv::Point, cv::Point>& points) noexcept
        {
            return
                std::sqrt
                (
                    square(points.first.x - points.second.x) +
                    square(points.first.y - points.second.y)
                );
        }
        [[nodiscard]] auto are_roughly_equal
            (const std::pair<std::size_t, std::size_t>& values, const std::size_t play) noexcept
        {
            static constexpr auto maximum_value = std::numeric_limits<decltype(values.first)>::max();

            return
                (values.first < play ? 0 : values.first - play) <= values.second &&
                (maximum_value - play < values.first ? maximum_value : values.first + play) >= values.second;
        }
        [[nodiscard]] auto is_parallelogram(const std::vector<cv::Point>& points) -> bool
        {
            static constexpr auto play = 0x28;

            return
                are_roughly_equal
                    ({distance({points[0], points[1]}), distance({points[3], points[2]})}, play) &&
                are_roughly_equal
                    ({distance({points[1], points[3]}), distance({points[2], points[0]})}, play);
        }

        template<typename To, typename From, std::size_t size>
        [[nodiscard]] auto convert_array(const std::array<From, size>& array) noexcept
            -> std::array<To, size>
        {
            std::array<To, size> to;
            std::transform
            (
                array.cbegin(),
                array.cend(),
                to.begin(),
                [](const auto element) { return static_cast<To>(element); }
            );

            return to;
        }
        [[nodiscard]] auto fit_to_frame(const Dimensions frame, const Dimensions image) noexcept
            -> FDimensions
        {
            return
                image.first > image.second ?
                FDimensions
                {
                    frame.first,

                    static_cast<float>(image.second) /
                    static_cast<float>(std::max(image.first, 1)) *
                    frame.first
                } :
                FDimensions
                {
                    static_cast<float>(image.first) /
                    static_cast<float>(std::max(image.second, 1)) *
                    frame.second,

                    frame.second,
                };
        }
    }


    // transform image to be ready for recognition function
    auto preprocess(const cv::Mat& image) -> cv::Mat
    {
        // blur strength
        static const cv::Size blur_size {3, 3};
        static constexpr auto blur_sigmax = 3;

        // canny sensitivity
        static constexpr auto canny_lower_threshold = 50;
        static constexpr auto canny_higher_threshold = 100;


        cv::Mat copy = image;

        // convert to grayscale
        cv::cvtColor(copy, copy, cv::COLOR_BGR2GRAY);

        // blur to clean up noise
        cv::GaussianBlur(copy, copy, blur_size, blur_sigmax);

        // detect prominent edges with "canny" algorithm
        cv::Canny(copy, copy, canny_lower_threshold, canny_higher_threshold);

        // dilate detected edges to close micro-gaps
        cv::dilate(copy, copy, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));

        return copy;
    }

    // find document in preprocessed image and return rectangle points in unspecified order
    [[nodiscard]] auto detect_document(const cv::Mat& image) -> Parallelogram
    {
        static constexpr auto minimum_area = 2'000;


        const auto contours =
            OUTPARAMETER
            (
                cv::findContours(image, OUT, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE),
                std::vector<std::vector<cv::Point>>
            );


        struct { double area; Parallelogram points; } biggest_rectangle;
        biggest_rectangle.area = 0;
        biggest_rectangle.points[0] = {0, 0};
        biggest_rectangle.points[1] = {0, 0};
        
        for (const auto& contour : contours)
        {
            const auto area = cv::contourArea(contour);

            // ignore shapes under noise threshold
            if (area < minimum_area)
            {
                continue;
            }

            const auto points =
                OUTPARAMETER
                (
                    cv::approxPolyDP(contour, OUT, 0.02 * cv::arcLength(contour, true), true),
                    std::vector<cv::Point>
                );

            // eliminate non-reactangles
            if (points.size() != 4)
            {
                continue;
            }

            if (area > biggest_rectangle.area && utility_math::is_parallelogram(points))
            {
                biggest_rectangle.area = area;
                biggest_rectangle.points = {points[0], points[1], points[2], points[3]};
            }
        }

        return biggest_rectangle.points;
    }

    // reorder points from document or any rectangle to prepare for warp
    auto reorder_points(const Parallelogram& points) -> Parallelogram
    {
        using PointValue = decltype(cv::Point::x);
        std::array<PointValue, 4> point_sums;
        std::array<PointValue, 4> point_differences;

        auto points_iterator = points.begin();
        auto point_sums_iterator = point_sums.begin();
        auto point_differences_iterator = point_differences.begin();

        while (points_iterator != points.end())
        {
            *point_sums_iterator++ = points_iterator->x + points_iterator->y;
            *point_differences_iterator++ = points_iterator->x - points_iterator->y;

            ++points_iterator;
        }

        return
            {
                {
                    points[std::min_element(point_sums.begin(), point_sums.end()) - point_sums.begin()],
                    points[std::max_element(point_differences.begin(), point_differences.end()) - point_differences.begin()],
                    points[std::min_element(point_differences.begin(), point_differences.end()) - point_differences.begin()],
                    points[std::max_element(point_sums.begin(), point_sums.end()) - point_sums.begin()]
                }
            };
    }

    // annotate found document in image with connected circles on given points
    [[nodiscard]] auto warp(const cv::Mat& image, const Parallelogram& points) -> cv::Mat
    {
        const auto detected_width = utility_math::distance({points[0], points[1]});
        const auto detected_height = utility_math::distance({points[0], points[2]});

        const auto dimensions =
            utility_math::fit_to_frame({image.cols, image.rows}, {detected_width, detected_height});

        const auto width = dimensions.first;
        const auto height = dimensions.second;

        const std::array<cv::Point2f, 4> source = utility_math::convert_array<cv::Point2f>(points);

        const std::array<cv::Point2f, 4> destination
        {
            {
                {0, 0}, {width, 0}, {0, height}, {width, height}
            }
        };

        return 
            OUTPARAMETER
            (
                cv::warpPerspective
                (
                    image,
                    OUT,
                    cv::getPerspectiveTransform(source, destination),
                    {static_cast<int>(width), static_cast<int>(height)}
                ),
                cv::Mat
            );
    }
}