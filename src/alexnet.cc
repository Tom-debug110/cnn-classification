#include<iostream>
#include<architectures.hpp>

cnn::architectures::AlexNet::AlexNet(const int numOfClasses, const bool batchNorm) {

    //TODO 第一层 卷积层 + ReLU + BatchNorm
    // batchSize x 3x224x224 ---> batchSiz * 16*111*111
    this->layerSequence_.emplace_back(new cnn::architectures::Conv2D("conv_layer_1", 3, 16, 3));
    if (batchNorm) {
        this->layerSequence_.emplace_back(std::make_shared<BatchNorm2D>("vn_layer_1", 16));
    }
    this->layerSequence_.emplace_back(std::make_shared<ReLU>("relu_layer_1"));

    //TODO 第二层  最大池化层
    // batchSize x16x111x111 ---> batchSize * 16*55*55
    this->layerSequence_.emplace_back(std::make_shared<MaxPool2D>("max_poll_1", 2, 2));

//    //TODO 第三层 卷积层 + ReLU + BatchNorm
//    // batchSize x16x55x55 ---> batchSize *32*27*27
//    this->layerSequence_.emplace_back(std::make_shared<Conv2D>("conv_layer_2", 16, 32, 3));
//    if (batchNorm) {
//        this->layerSequence_.emplace_back(std::make_shared<BatchNorm2D>("vn_layer_2", 32));
//    }
//    this->layerSequence_.emplace_back(std::make_shared<ReLU>("relu_layer_2"));
//
//    //TODO 第四层 卷积层 + ReLU + BatchNorm
//    // batchSize x32x27x27 ---> batchSize*64*13*13
//    this->layerSequence_.emplace_back(std::make_shared<Conv2D>("conv_layer_3", 32, 64, 3));
//    if (batchNorm) {
//        this->layerSequence_.emplace_back(std::make_shared<BatchNorm2D>("vn_layer_3", 64));
//    }
//    this->layerSequence_.emplace_back(std::make_shared<ReLU>("relu_layer_3"));
//
//    //TODO 第五层 卷积层 + ReLU + BatchNorm
//    // batchSize x64*13*13 ---> batchSize*128*6*6
//    this->layerSequence_.emplace_back(std::make_shared<Conv2D>("conv_layer_4", 64, 128, 3));
//    if (batchNorm) {
//        this->layerSequence_.emplace_back(std::make_shared<BatchNorm2D>("vn_layer_4", 128));
//    }
//    this->layerSequence_.emplace_back(std::make_shared<ReLU>("relu_layer_4"));

    //TODO 线性连接层
    // batchSize *128*6*6 ---> batchSize * numOfClasses
    this->layerSequence_.emplace_back(std::make_shared<LinearLayer>("linear_1", 16 * 2 * 2, numOfClasses));
}

std::vector<cnn::tensor> cnn::architectures::AlexNet::forward(const std::vector<tensor> &input) {
    assert(input.size());
    if (this->printInfo) {
        input.front()->printShape();
    }

    std::vector<tensor> output(input);
    int i = 0;
    for (const auto &sequence: layerSequence_) {
//        long long start = std::chrono::steady_clock::now().time_since_epoch().count();
//        cnn::architectures::printTensor(output);
        output = sequence->forward(output);
//        long long end = std::chrono::steady_clock::now().time_since_epoch().count();

//        std::cout << i++ << "  ---  " << end - start << std::endl;
        if (this->printInfo) {
            output.front()->printShape();
        }
    }
    return output;
}

void cnn::architectures::AlexNet::backward(std::vector<tensor> &delta) {
    if (this->printInfo) {
        delta.front()->printShape();
    }

    for (auto layer = layerSequence_.rbegin(); layer != layerSequence_.rend(); layer++) {
        delta = layer.operator->()->operator->()->backward(delta);
        if (this->printInfo) {
            delta.front()->printShape();
        }
    }
}

void cnn::architectures::AlexNet::updateGradients(const cnn::dataType learningRate) {
    for (const auto &layer: layerSequence_) {
        layer->updateGradients(learningRate);
    }
}

void cnn::architectures::AlexNet::saveWeights(const std::filesystem::path &path) const {
    // 只有 Conv2D LinearLayer BatchNorm2D 需要保存权重
    std::ofstream writer(path, std::ios::binary);
    for (const auto &layer: layerSequence_) {
        layer->saveWeights(writer);
    }

    std::cout << "weights have been saved to " << path.string() << std::endl;
    writer.close();
}

void cnn::architectures::AlexNet::loadWeights(const std::filesystem::path &path) {
    if (!std::filesystem::exists(path)) {
        std::cout << "预训练权重文件  " << path.string() << " 不存在 \n";
        return;
    }

    std::ifstream reader(path, std::ios::binary);
    for (const auto &layer: layerSequence_) {
        layer->loadWeights(reader);
    }

    std::cout << "load weights from " << path.string() << std::endl;

    reader.close();
}

cv::Mat cnn::architectures::AlexNet::gradCam(const std::string &layerName) const {
    return cv::Mat();
}