//
// Created by supermc on 2020/11/22.
//

#ifndef YOLOV5_CREATEENGINE_H
#define YOLOV5_CREATEENGINE_H

#include <iostream>
#include <chrono>
#include <sys/stat.h>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

#define NET bdd  // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) +
                               1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

ICudaEngine *createEngine_s(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt);

ICudaEngine *createEngine_m(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt);

ICudaEngine *createEngine_l(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt);

ICudaEngine *createEngine_x(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt);

ICudaEngine *createEngine_m_9(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt);
#include "createEngine_s.tpp"
#include "createEngine_m.tpp"
#include "createEngine_l.tpp"
#include "createEngine_x.tpp"
#include "createEngine_m_9.tpp"
#endif //YOLOV5_CREATEENGINE_H
