//
// Created by supermc on 2020/11/22.
//
#include "createEngine.hpp"

// Creat the engine using only the API and not any parser.

ICudaEngine *createEngine_x(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt) {
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../wts/yolov5x.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, 80, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 160, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 160, 160, 4, true, 1, 0.5,
                                         "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 320, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 320, 320, 12, true, 1, 0.5,
                                         "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 640, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 640, 640, 12, true, 1, 0.5,
                                         "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 1280, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 1280, 1280, 5, 9, 13, "model.8");

    /* ------- yolov5 head ------- */
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 1280, 1280, 4, false, 1, 0.5,
                                         "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 640, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * 640 * 2 * 2));
    for (int i = 0; i < 640 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{DataType::kFLOAT, deval, 640 * 2 * 2};
    IDeconvolutionLayer *deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 640, DimsHW{2, 2}, deconvwts11,
                                                                emptywts);
    deconv11->setStrideNd(DimsHW{2, 2});
    deconv11->setNbGroups(640);
    weightMap["deconv11"] = deconvwts11;

    ITensor *inputTensors12[] = {deconv11->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat12 = network->addConcatenation(inputTensors12, 2);

    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 1280, 640, 4, false, 1, 0.5,
                                          "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 320, 1, 1, 1, "model.14");

    Weights deconvwts15{DataType::kFLOAT, deval, 320 * 2 * 2};
    IDeconvolutionLayer *deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 320, DimsHW{2, 2}, deconvwts15,
                                                                emptywts);
    deconv15->setStrideNd(DimsHW{2, 2});
    deconv15->setNbGroups(320);
    ITensor *inputTensors16[] = {deconv15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 640, 320, 4, false, 1, 0.5,
                                          "model.17");

    // yolo layer 0
    IConvolutionLayer *det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5),
                                                        DimsHW{1, 1}, weightMap["model.24.m.0.weight"],
                                                        weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 320, 3, 2, 1, "model.18");
    ITensor *inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 640, 640, 4, false, 1, 0.5,
                                          "model.20");
    // yolo layer 1
    IConvolutionLayer *det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5),
                                                        DimsHW{1, 1}, weightMap["model.24.m.1.weight"],
                                                        weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 640, 3, 2, 1, "model.21");
    ITensor *inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 1280, 1280, 4, false, 1, 0.5,
                                          "model.23");
    // yolo layer 2
    IConvolutionLayer *det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5),
                                                        DimsHW{1, 1}, weightMap["model.24.m.2.weight"],
                                                        weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap) {
        free((void *) (mem.second.values));
    }

    return engine;
}