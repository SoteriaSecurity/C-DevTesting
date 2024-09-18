#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include </Users/omkarbantanur/Downloads/onnxruntime/include/onnxruntime_cxx_api.h>
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
    logging::add_file_log(
        keywords::file_name = "log/soteria_%Y-%m-%d.log",  // Log filename with date suffix
        keywords::rotation_size = 10 * 1024 * 1024,        // 10 MB rotation
        keywords::time_based_rotation = boost::log::sinks::file::rotation_at_time_point(0, 0, 0),  // Rotate at midnight
        keywords::format = "[%TimeStamp%] [%Severity%] %Message%",  // Log format
        keywords::auto_flush = true
    );

    logging::add_console_log(
        std::cout,
        keywords::format = "[%TimeStamp%] [%Severity%] %Message%"
    );

    logging::add_common_attributes();
}

void setupWebcam(Ort::Session& session, Ort::Env& env, std::vector<std::string> classes) {
    init_logging();
    BOOST_LOG_TRIVIAL(info) << "Application started";

    if (!fs::exists("log")) {

        fs::create_directory("log");
        BOOST_LOG_TRIVIAL(info) << "Log directory created";

    }

    cv::VideoCapture capture(0);

    if (!capture.isOpened()) {

        BOOST_LOG_TRIVIAL(error) << "Unable to open camera";
        exit(1);
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // Get input and output names
    auto inputNameAllocated = session.GetInputNameAllocated(0, allocator);
    auto outputNameAllocated = session.GetOutputNameAllocated(0, allocator);
    const char* inputName = inputNameAllocated.get();
    const char* outputName = outputNameAllocated.get();

    size_t inputTensorSize = 1 * 3 * 320 * 320;
    std::vector<int64_t> inputShape = {1, 3, 320, 320};

    while (true) {

        cv::Mat frame;

        if (!capture.read(frame)) {
            BOOST_LOG_TRIVIAL(error) << "Failed to read from camera";
            break;
        }

        cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);

        if (blob.empty()) {
            BOOST_LOG_TRIVIAL(error) << "Failed to create blob from frame!";
            continue;  // Skip processing if the blob is invalid
        }

        if (inputTensorSize != blob.total()) {
            BOOST_LOG_TRIVIAL(error) << "Mismatch in tensor size and blob data.";
            exit(1);
        }

        float* blobData = reinterpret_cast<float*>(blob.data);

        if (!blobData) {
            BOOST_LOG_TRIVIAL(error) << "Blob data is null!";
            exit(1);  // Exit if the blob data pointer is invalid
        }

            // Create input tensor
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, blobData, inputTensorSize, inputShape.data(), inputShape.size());

            // Run the inference session
            std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, 1);

            // Get the output data (outputTensors.front() gives the first output)
            float* outputData = outputTensors.front().GetTensorMutableData<float>();

            float confidenceThreshold = 0.75;

            for (size_t i = 0; i < outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount(); i += 7) {

                float confidence = outputData[i + 2];

                if (confidence > confidenceThreshold) {

                    int classId = static_cast<int>(outputData[i + 1]);
                    std::string label = classes[classId];

                    if (label == "weapon" || label == "fire") {

                        int x = static_cast<int>(outputData[i + 3] * frame.cols);
                        int y = static_cast<int>(outputData[i + 4] * frame.rows);
                        int width = static_cast<int>(outputData[i + 5] * frame.cols);
                        int height = static_cast<int>(outputData[i + 6] * frame.rows);

                        cv::Rect box(x - width / 2, y - height / 2, width, height);
                        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
                        cv::putText(frame, label, cv::Point(x - width / 2, y - height / 2 - 10),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }

            char key = (char)cv::waitKey(1);
            if (key == 27) {
                BOOST_LOG_TRIVIAL(info) << "ESC key pressed. Exiting.";
                break;
            }
        }

        capture.release();
        cv::destroyAllWindows();
        BOOST_LOG_TRIVIAL(info) << "Application ended";
    }


int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, "/Users/omkarbantanur/Downloads/best.onnx", sessionOptions);

    std::vector<std::string> classNames = {"money", "knife", "monedero", "pistol", "smartphone", "tarjeta"};

    setupWebcam(session, env, classNames);
    return 0;
}

// Helper function to get class names from a file