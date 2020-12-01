#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include "c++/7/fstream"
#include "c++/7/map"
#include "c++/7/sstream"
#include "c++/7/vector"
#include "opencv2/opencv.hpp"
#include "dirent.h"
#include "NvInfer.h"
#include "yololayer.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;

cv::Mat preprocess_img(cv::Mat &img);

cv::Rect get_rect(cv::Mat &img, float bbox[4]);

float iou(float lbox[4], float rbox[4]);

bool cmp(const Yolo::Detection &a, const Yolo::Detection &b);

void nms(std::vector<Yolo::Detection> &res, float *output, float conf_thresh, float nms_thresh);

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file);

#include "common.tpp"

#endif

