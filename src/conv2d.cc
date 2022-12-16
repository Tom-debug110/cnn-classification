#include<architectures.hpp>

std::vector<cnn::tensor> cnn::architectures::Conv2D::forward(const std::vector<tensor> &input) {
    const int batchSize = input.size();
    const int previousWidth = input.front()->getWidth();
    const int previousHeight = input.front()->getHeight();
    const int length = previousWidth * previousHeight;

    const int curWidth = (previousWidth - kernelSize_ - 2 * padding_) / stride_ + 1;
    const int curHeight = (previousHeight - kernelSize_ - 2 * padding_) / stride_ + 1;

    const int radius = kernelSize_ / 2;// 自动向下取整


    std::tuple<uint32_t, uint32_t, uint32_t> shape = std::make_tuple(outChannels_, curHeight, curWidth);

    initForward(batchSize, shape, previousWidth);// 初始化相关

    //cnn::architectures::printTensor(this->weights_);
    // 如果要backward 则需要记录当前的输入
    if (!noGrad) {
        this->_input_ = input;
    }

    const int convBoundaryOfWeight = previousWidth - radius;
    const int convBoundaryOfHeight = previousHeight - radius;

    const int windowsLength = kernelSize_ * kernelSize_;
    const int outLength = curWidth * curHeight;
    const int *offset = this->offset_.data();

    // 首先每一张图像分开卷积
    for (int b = 0; b < batchSize; ++b) {
        dataType *src = input.at(b)->getData();

        // 每一个卷积核
        for (int oc = 0; oc < outChannels_; ++oc) {
            // 输出位置指针
            dataType *outPtr = this->output_.at(b)->getData() + oc * outLength;
            // 卷积核权重指针
            dataType *weightPtr = weights_.at(oc)->getData();

            int cnt = 0;
            for (int x = radius; x < convBoundaryOfHeight; x += stride_) {
                for (int y = radius; y < convBoundaryOfWeight; y += stride_) {
                    dataType sumValue = 0.f;
                    const int coord = x * previousWidth + y;
                    for (int ic = 0; ic < inChannels_; ++ic) {

                        const int start = ic * length + coord; // 当前像素点相对于本张图像的偏移
                        const int startOfWeight = ic * windowsLength;

                        for (int k = 0; k < windowsLength; ++k) {
                            sumValue += src[start + offset[k]] * weightPtr[startOfWeight + k];
                        }
                    }

                    sumValue += this->bias_.at(oc);
                    outPtr[cnt] = sumValue;
                    ++cnt;
                }
            }
        }
    }

    return this->output_;
}

std::vector<cnn::tensor> cnn::architectures::Conv2D::backward(std::vector<tensor> &delta) {
    // 获取回传的信息 forward 的输出是多大 delta 就是多大
    const int batchSize = delta.size();
    const int outHeight = delta.front()->getHeight();
    const int outWidth = delta.front()->getWidth();

    const int outLength = outHeight * outWidth;

    // 获取之前 forward 的输入特征
    const uint32_t height = _input_.front()->getHeight();
    const uint32_t width = _input_.front()->getWidth();
    const uint32_t length = height * width;

    initBackward(this->weightsGradients_, outChannels_, {inChannels_, kernelSize_, kernelSize_},
                 this->name_ + "_weight_gradients_");

    this->biasGradients_.assign(outChannels_, 0);

    // 先把之前的梯度全部清空
    for (int oc = 0; oc < outChannels_; ++oc) {
        this->weightsGradients_.at(oc)->setZero();
        this->biasGradients_.at(oc) = 0;
    }

    //TODO 先计算 weight 和 bias 的梯度
    calWeightAndBiasGradients(delta, outHeight, outWidth, height, width);


    // TODO 计算从输出到输入的梯度 delta_output
    initBackward(this->deltaOutput_, batchSize, {inChannels_, height, width}, this->name_ + "_delta_");
    // 清零
    for (int b = 0; b < batchSize; ++b) {
        this->deltaOutput_.at(b)->setZero();
    }

    calDeltaGradients(delta, outHeight, outWidth, height, width);

    return this->deltaOutput_;
}

void cnn::architectures::Conv2D::updateGradients(const cnn::dataType learningRate) {
    assert(!this->weightsGradients_.empty());
    for (int oc = 0; oc < outChannels_; ++oc) {
        dataType *weightPtr = this->weights_.at(oc)->getData();
        dataType *weightGradientPtr = this->weightsGradients_.at(oc)->getData();

        for (int i = 0; i < paramsForAKernel_; ++i) {
            weightPtr[i] -= learningRate * weightGradientPtr[i];
        }

        bias_.at(oc) -= learningRate * biasGradients_.at(oc);
    }
}

void cnn::architectures::Conv2D::saveWeights(std::ofstream &writer) {
    // 需要保存的是 weights, bias
    const int filter_size = sizeof(dataType) * paramsForAKernel_;
    for (int o = 0; o < outChannels_; ++o)
        writer.write(reinterpret_cast<const char *>(&weights_[o]->getData()[0]),
                     static_cast<std::streamsize>(filter_size));
    writer.write(reinterpret_cast<const char *>(&bias_[0]),
                 static_cast<std::streamsize>(sizeof(dataType) * outChannels_));
}

void cnn::architectures::Conv2D::loadWeights(std::ifstream &reader) {
    const int filter_size = sizeof(dataType) * paramsForAKernel_;
    for (int o = 0; o < outChannels_; ++o)
        reader.read((char *) (&weights_[o]->getData()[0]),
                    static_cast<std::streamsize>(filter_size));
    reader.read((char *) (&bias_[0]),
                static_cast<std::streamsize>(sizeof(dataType) * outChannels_));
}

int cnn::architectures::Conv2D::getParamsNum() const {
    return (this->paramsForAKernel_ + 1) * this->outChannels_;
}

void
cnn::architectures::Conv2D::initForward(int batchSize, std::tuple<uint32_t, uint32_t, uint32_t> &shape, int preWeight) {
    if (this->output_.empty()) {
        this->output_.reserve(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            this->output_.emplace_back(new Tensor3D(shape,
                                                    this->name_ + "_output_" + std::to_string(i)));
        }
    }

    int radius = kernelSize_ / 2;

    int position = 0;
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            this->offset_[position++] = i * preWeight + j;
        }
    }

}

void
cnn::architectures::Conv2D::initBackward(std::vector<tensor> &v, int size,
                                         std::tuple<uint32_t, uint32_t, uint32_t> shape, std::string name) {
    if (v.empty()) {
        v.reserve(size);
        for (int i = 0; i < size; ++i) {
            v.emplace_back(std::make_shared<Tensor3D>(shape, name + std::to_string(i)));
        }
    }
}

void
cnn::architectures::Conv2D::calWeightAndBiasGradients(std::vector<tensor> &delta, uint32_t outHeight, uint32_t outWidth,
                                                      uint32_t height,
                                                      uint32_t width) {
    const uint32_t batchSize = delta.size();
    for (int b = 0; b < batchSize; ++b) {
        // 每一个卷积核
        for (int oc = 0; oc < outChannels_; ++oc) {
            dataType *outDelta = delta.at(b)->getData() + oc * outHeight * outWidth;

            //每一个输入的通道
            for (int ic = 0; ic < inChannels_; ++ic) {
                // 第 b 张输入，找到第 i 个通道的起始地址
                dataType *srcPtr = this->_input_.at(b)->getData() + ic * height * width;
                // 第 oc 个卷积核的第 i 个通道的起始地址
                dataType *weightPtr = this->weightsGradients_.at(oc)->getData() + ic * kernelSize_ * kernelSize_;

                //遍历卷积核中的每一个参数
                for (int kx = 0; kx < kernelSize_; ++kx) {
                    for (int ky = 0; ky < kernelSize_; ++ky) {
                        dataType sumValues = 0.f;

                        for (int x = 0; x < outHeight; ++x) {
                            dataType *deltaPtr = outDelta + x * outWidth;
                            dataType *inputPtr = srcPtr + (x * stride_ + kx) + width;

                            for (int y = 0; y < outWidth; ++y) {
                                // 当前的 weight 的梯度 由参与计算的输入和下一层返回的梯度相乘再累加
                                sumValues += deltaPtr[y] * inputPtr[y * stride_ + ky];
                            }
                        }
                        weightPtr[kx * kernelSize_ + ky] += sumValues / batchSize * 1.0;
                    }
                }
            }
            // 计算 bias 的梯度 bias 的大小就是 outChannels
            dataType sumValue = 0.f;
            for (int d = 0; d < outHeight * outWidth; ++d) {
                sumValue += outDelta[d];
            }

            biasGradients_[oc] += sumValue / batchSize * 1.0;
        }
    }
}

void cnn::architectures::Conv2D::calDeltaGradients(std::vector<tensor> &delta, uint32_t height, uint32_t width,
                                                   uint32_t inHeight, uint32_t inWidth) {
    const int batchSize = delta.size();
    const int radius = kernelSize_ / 2;
    const int boundaryHeight = height - radius;
    const int boundaryWidth = width - radius;
    const int windowsSize = kernelSize_ * kernelSize_;

    //  多个batch 分开计算
    for (int b = 0; b < batchSize; ++b) {
        // 输出  inChannels * 224 * 224
        dataType *deltaOut = this->deltaOutput_.at(b)->getData();
        for (int oc = 0; oc < outChannels_; ++oc) {
            dataType *outPtr = delta.at(b)->getData() + oc * height * width;
            dataType *weightPtr = this->weights_.at(oc)->getData();

            int cnt = 0;
            // 遍历图像平面上的每一个点
            for (int x = radius; x < inHeight - radius; x += stride_) {
                for (int y = 0; y < inWidth - radius; y += stride_) {
                    const int coord = x * width + y;
                    for (int ic = 0; ic < inChannels_; ++ic) {
                        const int start = ic * inHeight * inWidth + coord;
                        const int weightStart = ic * windowsSize;
                        for (int k = 0; k < windowsSize; ++k) {
                            deltaOut[start + offset_[k]] += weightPtr[weightStart + k] * outPtr[cnt];
                        }
                    }
                    cnt++;
                }
            }
        }
    }
}


