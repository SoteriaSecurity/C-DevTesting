#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

void* loadYolo() {
    cv::dnn::Net net = cv::dnn::readNet("fire_and_gun_detection/yolov3.weights", "fire_and_gun_detection/yolov3.cfg");
    std::vector<std::string> classes = {"Gun", "Fire", "Rifle"};

    std::vector<std::string> layerNames = net.getLayerNames();
    std::vector<std::string> outputLayers;
    std::vector<int> outLayers = net.getUnconnectedOutLayers();


    for (int i : outLayers) {
        outputLayers.push_back(layerNames[i - 1]);
    }

    std::vector<std::vector<float>> colors;

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < classes.size(); ++i) {
        std::vector<float> color = {
            static_cast<float>(std::rand() % 256),  // Red
            static_cast<float>(std::rand() % 256),  // Green
            static_cast<float>(std::rand() % 256)   // Blue
        };
        colors.push_back(color);
    }
    void *netPtr = &net, *classesPtr = &classes, *outputLayersPtr = &outputLayers, *colorsPtr = &colors;

    return netPtr, classesPtr, outputLayersPtr, colorsPtr; // RETURNS POINTERS TO EACH VALUE NOT THE ACTUAL VALUE
}


void* loadImage(std::string path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
    }

    double fx, fy;
    cv::resize(img, img, cv::Size(), 0.4, 0.4);

    int width = img.cols, height = img.rows, channels = img.channels();
    void *widthPtr = &width, *heightPtr = &height, *channelsPtr = &channels;

    return widthPtr, heightPtr, channelsPtr;
}
/*
void displayBlob(cv::Mat blob) {
    int num = blob.size[0];

    for (int b = 0; b < num; b++) {
        cv::Mat img = blob(cv::Range(b, b + 1), cv::Range::all(), cv::Range::all(), cv::Range::all()).clone();
        img = img.reshape(1, img.size[2]); // Reshape to height x width x channels

        // Display the image
        std::string windowName = std::to_string(b);
        cv::imshow(windowName, img);
    }

}
*/

