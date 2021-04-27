#include <string>
#include <opencv2/opencv.hpp>
#include <aug/outparameter.hpp>


// windowing functionality and abstractions
namespace testing::windows
{
    struct Window
    {
        // Title is "std::string" because "cv::imshow" takes string and not "std::string_view"
        using Title = std::string;
        using Image = cv::Mat;
        using Status = bool;


        // identifier to address window
        Title title;

        // displayed image
        Image image = {};

        // determines update behavior
        Status is_running = true;
    };

    auto show(const Window& window) -> void
    {
        cv::imshow(window.title.data(), window.image);
    }
    auto update(Window& window) -> void
    {
        window.is_running = cv::getWindowProperty(window.title, cv::WND_PROP_AUTOSIZE) != -1;

        if (window.is_running)
        {
            show(window);
        }
    }
}