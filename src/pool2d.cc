#include<architectures.hpp>

std::vector<cnn::tensor> cnn::architectures::MaxPool2D::forward(const std::vector<tensor> &input) {

    int batchSize = input.size();
    //std::cout << "batchSize  " << __LINE__ << "  " << batchSize << std::endl;
    const auto &first = input.front();

    const uint32_t channels = first->getChannels();
    const uint32_t height = first->getHeight();
    const uint32_t width = first->getWidth();
    std::tuple<uint32_t, uint32_t, uint32_t> shape{channels, height, width};

    const uint32_t poolOutPutHeight = (height - kernelSize_ + 2 * padding_) / step_ + 1;
    const uint32_t poolOutPutWidth = (width - kernelSize_ + 2 * padding_) / step_ + 1;

    init(batchSize, shape, poolOutPutHeight, poolOutPutWidth);

    // 开始池化
    const uint32_t length = height * width;
    const uint32_t outLength = poolOutPutWidth * poolOutPutHeight;
    const uint32_t poolBoundaryOfHeight = height - kernelSize_;
    const uint32_t poolBoundaryOfWidth = width - kernelSize_;
    const int windowsLength = kernelSize_ * kernelSize_;

    for (int b = 0; b < batchSize; ++b) {
        for (int c = 0; c < channels; ++c) {
            dataType *src = input.at(b)->getData() + c * length;
            dataType *dst = output_.at(b)->getData() + c * outLength;
            int *maskPtr = this->mask_.at(b).data() + c * outLength;

            int cnt = 0;

            for (int x = 0; x <= poolBoundaryOfHeight; x += step_) {
                dataType *row = src + x * width;
                for (int y = 0; y <= poolBoundaryOfWidth; y += step_) {
                    dataType maxValue = row[y];
                    int index = 0;
                    for (int i = 0; i < windowsLength; ++i) {
                        dataType comp = row[y + offset_[i]];
                        //std::cout << comp << " \n";
                        if (comp > maxValue) {
                            maxValue = comp;
                            index = offset_[i];
                        }
                    }

                    // 局部最大值输出到对应位置
                    dst[cnt] = maxValue;

                    if (!noGrad) {
                        index += x * width + y;
                        maskPtr[cnt] = c * length + index;
                    }
                    cnt++;
                }
            }
        }
    }

    return this->output_;
}

std::vector<cnn::tensor> cnn::architectures::MaxPool2D::backward(std::vector<tensor> &delta) {
    // 获取输入的梯度的信息
    const int batchSize = delta.size();

    // 先对 setZero 清零处理 因为不提供最大值的部分的梯度都是零
    for (int b = 0; b < batchSize; ++b) {
        this->deltaOutput_.at(b)->setZero();
    }

    const int totalLength = delta.front()->length();
    for (int b = 0; b < batchSize; ++b) {
        int *maskPtr = this->mask_.at(b).data();

        dataType *src = delta[b]->getData();
        dataType *dst = deltaOutput_[b]->getData();

        for (int i = 0; i < totalLength; ++i) {
            dst[maskPtr[i]] = src[i];
        }
    }

    return deltaOutput_;
}

void cnn::architectures::MaxPool2D::init(int batchSize, std::tuple<uint32_t, uint32_t, uint32_t> shape, uint32_t height,
                                         uint32_t width) {
    if (!this->output_.empty()) {
        return;
    }
    //std::cout << "maxPool Init  "<<batchSize << std::endl;

    std::tuple<uint32_t, uint32_t, uint32_t> outShape{std::get<0>(shape), height, width};
    this->output_.reserve(batchSize);
    for (int i = 0; i < batchSize; ++i) {
        this->output_.emplace_back(std::make_shared<Tensor3D>(outShape, this->name_ + "_output_" + std::to_string(i)));
    }

    if (!noGrad) {
        this->deltaOutput_.reserve(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            this->deltaOutput_.emplace_back(
                    std::make_shared<Tensor3D>(shape, this->name_ + "_delta_" + std::to_string(i)));
        }

        // mask 对 batch 中的每一张图都分配空间
        int length = std::get<0>(shape) * height * width;
        this->mask_.reserve(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            this->mask_.emplace_back(length);
        }
    }

    int position = 0;
    for (int i = 0; i < kernelSize_; ++i) {
        for (int j = 0; j < kernelSize_; ++j) {
            this->offset_.at(position++) = i * std::get<2>(shape) + j;
        }
    }
}
