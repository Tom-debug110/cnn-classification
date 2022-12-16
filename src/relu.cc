#include<architectures.hpp>

std::vector<cnn::tensor> cnn::architectures::ReLU::forward(const std::vector<tensor> &input) {
    const int batchSize = input.size();
    auto shape = input.front()->shape();
    init(batchSize, shape);

    const int length = input.front()->Length();

    for (int i = 0; i < batchSize; ++i) {
        dataType *src = input.at(i)->getData();
        dataType *dst = this->output_.at(i)->getData();

        for (int j = 0; j < length; ++j) {
            dst[j] = (src[j] >= 0) ? src[j] : 0;
        }
    }

    return this->output_;
}

std::vector<cnn::tensor> cnn::architectures::ReLU::backward(std::vector<tensor> &delta) {
    // 反向传播不需要在分配空间啦
    // 同时ReLU 层是原地进行反向传播，也并不需要返回给上一层 delta 的输出

    const int batchSize = delta.size();
    const int length = delta.front()->Length();

    for (int i = 0; i < batchSize; ++i) {
        dataType *src = delta.at(i)->getData();
        dataType *out = this->output_.at(i)->getData();

        for (int j = 0; j < length; ++j) {
            src[j] = (out[j] <= 0) ? 0 : src[j];
        }
    }

    return delta;
}

void cnn::architectures::ReLU::init(int size, std::tuple<uint32_t, uint32_t, uint32_t> shape) {
    if (this->output_.empty()) {
        this->output_.reserve(size);

        for (int i = 0; i < size; ++i) {
            this->output_.emplace_back(std::make_shared<Tensor3D>(shape, this->name_ + "_output_" + std::to_string(i)));
        }
    }


}
