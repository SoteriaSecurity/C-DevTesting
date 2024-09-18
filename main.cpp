/*

#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <stdlib.h>
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

void setupWebcam(cv::dnn::Net net, std::vector<std::string> classes) {
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
        exit(1);
    }

    while (true) {

        cv::Mat frame;

        // Capture frame-by-frame
        bool frame_available = capture.read(frame);

        if (!frame_available) {
            BOOST_LOG_TRIVIAL(error) << "Failed to read from camera";
            break;
        }

        cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        float confidenceThreshold = 0.75;  // Confidence threshold

        for (const auto& output : outputs) {

            auto* data = (float*)output.data;

            for (int i = 0; i < output.rows; ++i, data += output.cols) {

                float confidence = data[4];  // Objectness score

                if (confidence > confidenceThreshold) {

                    float* classesScores = data + 5;
                    cv::Point classIdPoint;
                    double maxClassScore;

                    cv::minMaxLoc(cv::Mat(1, classes.size(), CV_32FC1, classesScores), 0, &maxClassScore, 0, &classIdPoint);

                    if (maxClassScore > confidenceThreshold) {
                        int classId = classIdPoint.x;
                        std::string label = classes[classId];

                        // Only process "weapon" or "fire" classes
                        if (label == "weapon" || label == "fire") {
                            // Extract bounding box coordinates
                            int x = static_cast<int>(data[0] * frame.cols);
                            int y = static_cast<int>(data[1] * frame.rows);
                            int width = static_cast<int>(data[2] * frame.cols);
                            int height = static_cast<int>(data[3] * frame.rows);

                            // Draw bounding box and label
                            cv::Rect box(x - width / 2, y - height / 2, width, height);
                            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
                            cv::putText(frame, label, cv::Point(x - width / 2, y - height / 2 - 10),
                                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                        }
                    }
                }
            }
        }
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
}

// Helper function to get class names from a file
/*
 std::vector<std::string> getClassNames(const std::string& filename) {
    std::vector<std::string> classNames;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening class names file!" << std::endl;
        return classNames;
    }
    std::string line;
    while (std::getline(file, line)) {
        classNames.push_back(line);
    }
    return classNames;
}


int main() {


    std::string modelWeights = "best.onnx";
    std::vector<std::string> classNames = {"money", "knife", "monedero", "pistol", "smartphone", "tarjeta"};

    cv::dnn::Net net = cv::dnn::readNetFromONNX( modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

    setupWebcam(net, classNames);


    return 0;
}

*/

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    const std::string modelWeights = "/Users/omkarbantanur/Downloads/best.onnx";

    // Try loading the ONNX model
    cv::dnn::Net net;
    try {
        net = cv::dnn::readNetFromONNX(modelWeights);
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading ONNX model: " << modelWeights << std::endl;
        std::cerr << e.what() << std::endl;
        return -1;
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

    std::cout << "Model loaded successfully!" << std::endl;
    return 0;
}
