#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#define OPENCV
#define TRACK_OPTFLOW
#define GPU

#include "net.h"
#include "benchmark.h"
#include <vector>
#include <stdio.h>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <fstream>
#include <regex>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>


struct ObjectTrace
{
    cv::Rect2f rect; // xmin, ymin, width, height
    std::string label;
    float score;
    int obj_id;   
};


class ObjectDetector {
public:
    ObjectDetector(std::string& objNameFile, std::string& cfgFile, std::string& weightFile);
    ~ObjectDetector();
    void processDetect(const cv::Mat& frame, float thresh, std::vector<ObjectTrace>& detectResult);
    std::vector<std::string> objNames;

private:
    int init(const char* model_bin, const char* model_param);
    std::vector<std::string> loadObjectNames(const std::string& filename);
    int width_models = 0;
    int height_models = 0;
    ncnn::Net* ncnnNet;
    ncnn::UnlockedPoolAllocator* g_blob_pool_allocator_detect;
    ncnn::PoolAllocator* g_workspace_pool_allocator_detect;

};

#endif /* ObjectDetector_hpp */
