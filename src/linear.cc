#include<architectures.hpp>

std::vector<cnn::tensor> cnn::architectures::LinearLayer::forward(const std::vector<tensor> &input) {
    // 线性层前向传播
    const int batchSize = input.size();
    this->deltaShape_ = input.front()->shape();

    // 开始 forward 之前清空 output, 再进行重新填充
    std::vector<tensor>().swap(this->output_);
    for (int b = 0; b < batchSize; ++b) {
        this->output_.emplace_back(
                std::make_shared<Tensor3D>(outChannels_, this->name_ + "_output_" + std::to_string(b)));
    }

    // 反向传播需要
    if (!noGrad) {
        this->_input_ = input;
    }


    for (int b = 0; b < batchSize; ++b) {
        dataType *srcPtr = input.at(b)->getData();
        dataType *outPtr = this->output_.at(b)->getData();

        for (int oc = 0; oc < outChannels_; ++oc) {
            dataType sumValue = 0;
            for (int ic = 0; ic < inChannels_; ++ic) {
                sumValue += srcPtr[ic] * weights_[ic * outChannels_ + oc];
            }

            outPtr[oc] = sumValue;

        }
    }
    return std::vector<tensor>{this->output_};
}

std::vector<cnn::tensor> cnn::architectures::LinearLayer::backward(std::vector<tensor> &delta) {
    const int batchSize = delta.size();
    if (this->weightGradients_.empty()) {
        this->weightGradients_.assign(inChannels_ * outChannels_, 0);
        this->biasGradients_.assign(outChannels_, 0);
    }

    calWeightGradients(delta);
    calBiasGradients(delta);

    if (deltaOutPut_.empty()) {
        this->deltaOutPut_.reserve(batchSize);
        for (int b = 0; b < batchSize; ++b) {
            this->deltaOutPut_.emplace_back(
                    std::make_shared<Tensor3D>(static_cast<std::tuple<uint32_t, uint32_t, uint32_t> &>(deltaShape_),
                                               "linear_delta_" + std::to_string(b)));
        }
    }

    calInputGradients(delta);

    return this->deltaOutPut_;
}

void cnn::architectures::LinearLayer::updateGradients(const cnn::dataType learningRate) {
    assert(!this->weightGradients_.empty());
    assert(!this->biasGradients_.empty());

    const int length = inChannels_ * outChannels_;
    for (int i = 0; i < length; ++i) {
        this->weights_[i] -= learningRate * this->weightGradients_[i];
    }

    for (int i = 0; i < outChannels_; ++i) {
        this->bias_[i] -= learningRate * this->biasGradients_[i];
    }
}

void cnn::architectures::LinearLayer::saveWeights(std::ofstream &writer) {
    writer.write(reinterpret_cast<const char *>(&weights_[0]),
                 static_cast<std::streamsize>(sizeof(dataType) * inChannels_ * outChannels_));
    writer.write(reinterpret_cast<const char *>(&bias_[0]),
                 static_cast<std::streamsize>(sizeof(dataType) * outChannels_));
}

void cnn::architectures::LinearLayer::loadWeights(std::ifstream &reader) {
    reader.read((char *) (&weights_[0]), static_cast<std::streamsize>(sizeof(dataType) * inChannels_ * outChannels_));
    reader.read((char *) (&bias_[0]), static_cast<std::streamsize>(sizeof(dataType) * outChannels_));
}

void cnn::architectures::LinearLayer::calWeightGradients(std::vector<tensor> &delta) {
    int batchSize = delta.size();
    for (int ic = 0; ic < inChannels_; ++ic) {
        dataType *weightGradientsPtr = weightGradients_.data() + ic * outChannels_;
        for (int oc = 0; oc < outChannels_; ++oc) {
            dataType sumValue = 0;
            for (int b = 0; b < batchSize; ++b) {
                sumValue += this->_input_.at(b)->getData()[ic] * delta.at(b)->getData()[oc];
            }
            weightGradientsPtr[oc] = sumValue / randomTimes;
        }
    }
}

void cnn::architectures::LinearLayer::calBiasGradients(std::vector<tensor> &delta) {
    int batchSize = delta.size();
    for (int oc = 0; oc < outChannels_; ++oc) {
        dataType sumValue = 0;
        for (int b = 0; b < batchSize; ++b) {
            sumValue += delta.at(b)->getData()[oc];
        }
        this->biasGradients_[oc] = sumValue / batchSize;
    }
}

void cnn::architectures::LinearLayer::calInputGradients(std::vector<tensor> &delta) {

    int batchSize = delta.size();
    for (int b = 0; b < batchSize; ++b) {
        dataType *deltaPtr = delta.at(b)->getData();
        dataType *outPtr = deltaOutPut_.at(b)->getData();

        for (int ic = 0; ic < inChannels_; ++ic) {
            dataType sumValue = 0;
            dataType *weightPtr = this->weights_.data() + ic * outChannels_;
            for (int oc = 0; oc < outChannels_; ++oc) {
                sumValue += deltaPtr[oc] * weightPtr[oc];
            }
            outPtr[ic] = sumValue;
        }
    }
}

