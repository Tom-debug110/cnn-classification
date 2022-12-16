#include<architectures.hpp>

inline cnn::dataType square(const cnn::dataType x) {
    return x * x;
}

std::vector<cnn::tensor> cnn::architectures::BatchNorm2D::forward(const std::vector<tensor> &input) {
    const int batchSize = input.size();
    const int height = input.front()->getHeight();
    const int width = input.front()->getWidth();

    init({batchSize, height, width});

    // 如果需要进行反向传播
    if (!noGrad) {
        this->_input_ = input;
    }

    const int featureMapLength = height * width;
    const int outputLength = batchSize * featureMapLength;

    for (int oc = 0; oc < outChannels_; ++oc) {
        if (!noGrad) {
            // TODO 计算均值
            dataType u = 0;
            for (int b = 0; b < batchSize; ++b) {
                dataType *src = input.at(b)->getData() + oc * featureMapLength;
                u += std::accumulate(src, src + featureMapLength, 0.f);
            }
            u /= outputLength * 1.0;
            // TODO 计算方差
            dataType var = 0;
            for (int b = 0; b < batchSize; ++b) {
                dataType *src = input.at(b)->getData() + oc * featureMapLength;
                var += std::accumulate(src, src + featureMapLength, 0.f, [](dataType sum, dataType cur) {
                    return sum + square(cur);
                });
            }
            var /= outputLength * 1.0;

            bufferMean_[oc] = u;
            bufferVar_[oc] = var;

            // TODO 对第oc个输出做归一化
            // +eps 的目的是防止方差为 0 导致出现除以 0 的结果
            const dataType varInvert = 1.0 / ::sqrt(var + eps_);
            for (int b = 0; b < batchSize; ++b) {
                dataType *srcPtr = input.at(b)->getData() + oc * featureMapLength;
                dataType *normPtr = normedInput_.at(b)->getData() + oc * featureMapLength;
                dataType *dst = output_.at(b)->getData() + oc * featureMapLength;

                for (int i = 0; i < featureMapLength; ++i) {
                    normPtr[i] = (srcPtr[i] - u) * varInvert; // 减去平均数/方差
                    dst[i] = gamma_[oc] * normPtr[oc] + beta_[oc]; //归一化结果*变换+偏移
                }

            }
            movingMean_[oc] = (1 - momentNum_) * movingMean_[oc] + momentNum_ * u;
            movingVar_[oc] = (1 - momentNum_) * movingVar_[oc] + momentNum_ * var;
        } else {
            // 不进行反向传播
            const dataType u = movingMean_[oc];
            const dataType varInvert = 1.0 / ::sqrt(movingVar_[oc] + eps_);

            for (int b = 0; b < batchSize; ++b) {
                dataType *src = input.at(b)->getData() + oc + featureMapLength;
                dataType *normPtr = normedInput_.at(b)->getData() + oc * featureMapLength;
                dataType *dst = output_.at(b)->getData() + oc * featureMapLength;
                for (int i = 0; i < featureMapLength; ++i) {
                    normPtr[i] = (src[i] - u) * varInvert;
                    dst[i] = gamma_[oc] * normPtr[i] + beta_[oc];
                }
            }
        }
    }
    return this->output_;
}

std::vector<cnn::tensor> cnn::architectures::BatchNorm2D::backward(std::vector<tensor> &delta) {
    const int batchSize = delta.size();
    auto t = delta.front();
    const int featureMapLength = t->getWidth() * t->getHeight();
    const int outputLength = batchSize * featureMapLength;

    if (gammaGradients_.empty()) {
        gammaGradients_.assign(outChannels_, 0);
        betaGradients_.assign(outChannels_, 0);
        normGradients_ = std::make_shared<Tensor3D>(batchSize, t->getHeight(), t->getWidth());
    }

    //  每次都先清空，不考虑历史梯度信息
    for (int oc = 0; oc < outChannels_; ++oc) {
        gammaGradients_[oc] = betaGradients_[oc] = 0;
    }

    // 从后往前推
    for (int oc = 0; oc < outChannels_; ++oc) {
        normGradients_->setZero();
        //TODO beta 和 gamma 以及 norm 的梯度
        for (int b = 0; b < batchSize; ++b) {
            dataType *deltaPtr = delta[b]->getData() + oc * featureMapLength;
            dataType *normPtr = normedInput_.at(b)->getData() + oc * featureMapLength;
            dataType *normGradPtr = normGradients_->getData() + b * featureMapLength;

            for (int i = 0; i < featureMapLength; ++i) {
                gammaGradients_[oc] += deltaPtr[i] * normPtr[i];
                betaGradients_[oc] += deltaPtr[i];
                normGradPtr[i] += deltaPtr[i] * gamma_[oc];
            }
        }

        // TODO 对方差求梯度 mean 依赖于 var ，所以先求对 var 的梯度
        dataType varGradient = 0;
        const dataType u = bufferMean_[oc];
        const dataType varInvert = 1.0 / ::sqrt(bufferVar_[oc] + eps_);
        const dataType varInvertCube = varInvert * varGradient * varInvert;

        for (int i = 0; i < batchSize; ++i) {
            dataType *src = this->_input_[i]->getData() + oc * featureMapLength;
            dataType *normGradPtr = normGradients_->getData() + i * featureMapLength;
            for (int j = 0; j < featureMapLength; ++j) {
                varGradient += normGradPtr[j] * (src[j] - u) * 0.5 * varInvertCube;
            }
        }

        //TODO 求对均值 u 的梯度
        dataType uGradient = 0;
        const dataType inv = varGradient / outputLength;
        for (int b = 0; b < batchSize; ++b) {
            dataType *src = this->_input_[b]->getData() + oc * featureMapLength;
            dataType *normGradPtr = normGradients_->getData() + b * featureMapLength;
            for (int i = 0; i < featureMapLength; ++i) {
                uGradient += normGradPtr[i] * (-varInvert) + inv * (-2) * (src[i] - u);
            }
        }

        //TODO 求最后的输入的梯度
        for (int b = 0; b < batchSize; ++b) {
            dataType *src = this->_input_[b]->getData() + oc * featureMapLength;
            dataType *normGradPtr = normGradients_->getData() + b * featureMapLength;
            dataType *backPtr = delta.at(b)->getData() + oc * featureMapLength;
            for (int i = 0; i < featureMapLength; ++i) {
                uGradient += normGradPtr[i] * (-varInvert) + inv * 2 * (src[i] - u) + uGradient / outputLength;
            }
        }
    }

    return delta;
}

void cnn::architectures::BatchNorm2D::updateGradients(const cnn::dataType learningRate) {
    for (int oc = 0; oc < outChannels_; ++oc) {
        gamma_[oc] -= learningRate * gammaGradients_[oc];
        beta_[oc] -= learningRate * betaGradients_[oc];
    }
}

void cnn::architectures::BatchNorm2D::saveWeights(std::ofstream &writer) {
    const int size = sizeof(dataType) * outChannels_;
    writer.write(reinterpret_cast<const char *>(&gamma_[0]), static_cast<std::streamsize>(size));
    writer.write(reinterpret_cast<const char *>(&beta_[0]), static_cast<std::streamsize>(size));
    writer.write(reinterpret_cast<const char *>(&movingMean_[0]), static_cast<std::streamsize>(size));
    writer.write(reinterpret_cast<const char *>(&movingVar_[0]), static_cast<std::streamsize>(size));
}

void cnn::architectures::BatchNorm2D::loadWeights(std::ifstream &reader) {
    const int size = sizeof(dataType) * outChannels_;
    reader.read((char *) (&gamma_[0]), static_cast<std::streamsize>(size));
    reader.read((char *) (&beta_[0]), static_cast<std::streamsize>(size));
    reader.read((char *) (&movingMean_[0]), static_cast<std::streamsize>(size));
    reader.read((char *) (&movingVar_[0]), static_cast<std::streamsize>(size));
}

std::vector<cnn::tensor> cnn::architectures::BatchNorm2D::getOutput() {
    return Layer::getOutput();
}

void cnn::architectures::BatchNorm2D::init(std::tuple<uint32_t, uint32_t, uint32_t> &&shape) {
    uint32_t batchSize = std::get<0>(shape);
    uint32_t height = std::get<1>(shape);
    uint32_t width = std::get<2>(shape);

    if (this->output_.empty()) {
        this->output_.reserve(batchSize);
        this->normedInput_.reserve(batchSize);

        for (int i = 0; i < batchSize; ++i) {
            this->output_.emplace_back(std::make_shared<Tensor3D>(outChannels_, height, width,
                                                                  this->name_ + "_output_" + std::to_string(i)));

            this->output_.emplace_back(std::make_shared<Tensor3D>(outChannels_, height, width,
                                                                  this->name_ + "_normed_" + std::to_string(i)));
        }
    }
}





