#include <iostream>
#include <array>
#include <string>

#include <aug/outparameter.hpp>  // functionality to capture outparameters
#include <aug/cast.hpp>  // only used for ignoring return value of functions

#include "types.hpp"
#include "processing.hpp"
#include "visuals.hpp"
#include "windows.hpp"


namespace testing::main
{
    // a pair of windows
    // of which one displays the input
    // and the other the output of a process
    using TransformationWindows = std::pair<testing::windows::Window, testing::windows::Window>;

    // combine previous algorithms
    auto process(cv::Mat& source, cv::Mat& warp) -> bool
    {
        const auto unordered_points =
            processing::detect_document(testing::processing::preprocess(source));

        // case with width 0 is impossible thus indicates that none was found
        // in that case make warp black screen
        static const cv::Point origin {0, 0};
        if (unordered_points[0] == origin && unordered_points[1] == origin)
        {
            return false;
        }

        const auto points = processing::reorder_points(unordered_points);
        warp = processing::warp(source, points);

        // visual feedback on what was detected
        visuals::annotate_document(source, unordered_points); 

        return true;
    }
    auto update(TransformationWindows& windows, cv::VideoCapture& capture) -> void
    {
        // read image from capture card (webcam)
        capture >> windows.first.image;

        if (testing::main::process(windows.first.image, windows.second.image))
        {
            update(windows.second);
        }
        update(windows.first);

        aug::cast::ignore(cv::waitKey(1));
    }
}


auto main() -> int
{
    // scale original size to fit screen
    static constexpr auto source_window_scalar = 1;
    static constexpr auto warp_window_scalar = 1.5;

    // titles/handles of output windows
    static const testing::windows::Window::Title source_window_title {"input"};
    static const testing::windows::Window::Title warp_window_title {"warp"};


    // open webcam #0
    cv::VideoCapture capture {0};

    // setup windows
    cv::namedWindow(source_window_title);
    cv::namedWindow(warp_window_title);

    testing::main::TransformationWindows windows
    {
        {source_window_title, source_window_scalar},
        {warp_window_title, warp_window_scalar}
    };


    // main loop

    // run until either the capture card is done or all windows are closed
    while (capture.isOpened() && windows.first.is_running)
    {
        testing::main::update(windows, capture);
    }

    // clean up resources
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}