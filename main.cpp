#include <filesystem>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

// Namespace for filesystem
namespace fs = std::filesystem;

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace keywords = boost::log::keywords;

void init_logging() {
    // Set log file format and rotate daily
    logging::add_file_log(
        keywords::file_name = "log/soteria_%Y-%m-%d.log",  // Log filename with date suffix
        keywords::rotation_size = 10 * 1024 * 1024,        // 10 MB rotation
        keywords::time_based_rotation = boost::log::sinks::file::rotation_at_time_point(0, 0, 0),  // Rotate at midnight
        keywords::format = "[%TimeStamp%] [%Severity%] %Message%",  // Log format
        keywords::auto_flush = true
    );

    // Add console output
    logging::add_console_log(
        std::cout,
        keywords::format = "[%TimeStamp%] [%Severity%] %Message%"
    );

    // Add common attributes like timestamp
    logging::add_common_attributes();
}

int main() {
    // Initialize logger
    init_logging();
    BOOST_LOG_TRIVIAL(info) << "Application started";

    // Create log directory if it doesn't exist
    if (!fs::exists("log")) {
        fs::create_directory("log");
        BOOST_LOG_TRIVIAL(info) << "Log directory created";
    }

    // Open the default camera (index 0)
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        BOOST_LOG_TRIVIAL(error) << "Unable to open camera";
        return -1;
    }

    cv::Mat frame;
    while (true) {
        // Capture frame-by-frame
        bool frame_available = capture.read(frame);
        if (!frame_available) {
            BOOST_LOG_TRIVIAL(error) << "Failed to read from camera";
            break;
        }

       cv::flip(frame, frame, 1);

        // Display the resulting frame
        cv::imshow("Webcam", frame);

        // Capture keyboard input
        char key = (char)cv::waitKey(1);

        // If 'ESC' key is pressed, exit loop
        if (key == 27) {
            BOOST_LOG_TRIVIAL(info) << "ESC key pressed. Exiting.";
            break;
        }
    }

    // Release the camera and close any OpenCV windows
    capture.release();
    cv::destroyAllWindows();

    BOOST_LOG_TRIVIAL(info) << "Application ended";
    return 0;
}
