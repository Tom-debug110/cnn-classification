#include<data_format.hpp>
#include<iomanip>

/**
 * @brief 从cv::Mat 中读取数据到 this->data_
 * @param image cv::Mat 图像
 */
void cnn::Tensor3D::readData(cv::Mat &image) {
    uint32_t size = this->width_ * this->height_;
    readData(image.data, size);
}

/**
 * @brief @brief 从原始指针读取数据
 * @param image *const uchar *const 类型指针
 * @param size 指针区域的大小
 */
void cnn::Tensor3D::readData(const uchar *const image, const uint32_t size) {

    // 最初是按照一行一行的进行复制，但是OpenCV里面本来也是线性的，还使用二维的方式反而多次一举啦
    for (int i = 0; i < size; ++i) {
        uint32_t position = 3 * i;
        this->data_[i] = image[position] * 1.0 / 255;
        this->data_[size + i] = image[position + 1] * 1.0 / 255;
        this->data_[size * 2 + i] = image[position + 2] * 1.0 / 255;
//        ::printf("%d %d %d\n", image[position], image[position + 1], image[position + 2]);
    }
}

/**
 * @brief 对 this->data_ 进行清零，使用 ::memset()实现
 */
void cnn::Tensor3D::setZero() {
    ::memset(this->data_, 0, sizeof(cnn::dataType) * this->length_);
}


/**
 * @brief 当前数据中的最大值
 * @return dataType 最大值
 */
cnn::dataType cnn::Tensor3D::max() const {
    return this->data_[argmax()];
}

/**
 * @brief 求得最大值的位置，也就是下标
 * @return 最大值下标
 */
uint32_t cnn::Tensor3D::argmax() const {
    dataType value = this->data_[0];
    uint32_t index = 0;
    for (int i = 1; i < length_; ++i) {
        if (this->data_[i] > value) {
            value = this->data_[i];
            index = i;
        }
    }

    return index;
}

/**
 * @brief 当前数据中的最小值
 * @return deteType 最小值
 */
cnn::dataType cnn::Tensor3D::min() const {
    return this->data_[argmin()];
}

uint32_t cnn::Tensor3D::argmin() const {
    dataType value = this->data_[0];
    uint32_t index = 0;
    for (int i = 1; i < length_; ++i) {
        if (this->data_[i] < value) {
            value = this->data_[i];
            index = i;
        }
    }

    return index;
}

/**
 * @brief 对所有数据进行缩放,即所有数据除以一个数
 * @param times 缩放倍数 dataItem / times
 */
void cnn::Tensor3D::div(const cnn::dataType times) {
    for (int i = 0; i < length_; ++i) {
        this->data_[i] /= times;
    }
}

/**
 * @brief 对数据进行归一化 采用(data-mean)/standardDeviation
 * @param mean 平均数
 * @param standardDeviation 标准差
 */
void cnn::Tensor3D::normalize(const std::vector<dataType> &mean, const std::vector<dataType> &standardDeviation) {
    if (this->channels_ != 3) {
        return;
    }

    for (int c = 0; c < channels_; ++c) {
        cnn::dataType *curChannel = this->data_ + c * length_;
        for (int i = 0; i < length_; ++i) {
            curChannel[i] = (curChannel[i] - mean.at(c)) / standardDeviation.at(c);
        }
    }
}

/**
 * @brief 将数据转换为 cv::Mat 的格式
 * @param CH 转换的通道数
 * @return
 */
cv::Mat cnn::Tensor3D::opencvMat(const int CH) const {
    int H = height_;
    int W = width_;
    // 只针对没有进行 normalize 的 Tensor 可以取出数据查看, 坑不填了, 懒得
    cv::Mat origin;
    if (CH == 3) {
        origin = cv::Mat(H, W, CV_8UC3);
        const int length = H * W;
        for (int i = 0; i < length; ++i) {
            const int p = 3 * i;
            origin.data[p] = cv::saturate_cast<uchar>(255 * data_[i]);
            origin.data[p + 1] = cv::saturate_cast<uchar>(255 * data_[i + length]);
            origin.data[p + 2] =
                    cv::saturate_cast<uchar>(255 * data_[i + length + length]);
        }
    } else if (CH == 1) {
        origin = cv::Mat(H, W, CV_8UC1);
        const int length = H * W;
        for (int i = 0; i < length; ++i)
            origin.data[i] = cv::saturate_cast<uchar>(255 * data_[i]);
    }
    return origin;
}

/**
 * @brief 当前存储数据的长度，即数据个数
 * @return
 */
uint32_t cnn::Tensor3D::length() const {
    return this->length_;
}

/**
 * @brief 挡墙 Tensor 的形状，即 width height 等参数
 * @return
 */
std::tuple<uint32_t, uint32_t, uint32_t> cnn::Tensor3D::shape() const {
    return {this->channels_, this->height_, this->width_};
}

void cnn::Tensor3D::printShape() const {
    ::printf("[%d,%d,%d]\n", this->channels_, this->height_, this->width_);
}

/**
 * @brief 输出指定通道的数据值
 * @param channel 通道
 */
void cnn::Tensor3D::print(uint32_t channel) const {
    if (channel > this->channels_) {
        return;
    }
    std::cout << this->name_ << " content: \n";
    const uint32_t index = (channel - 1) * this->width_ * this->height_;
    for (int h = 0; h < this->height_; ++h) {
        for (int w = 0; w < this->width_; ++w) {
            std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3)
                      << this->data_[index + h * this->width_ + w] << " ";
        }
        std::cout << "\n";
    }
}

/**
 * @brief 图像数据旋转180度
 * @return 旋转180度之后的Tensor
 */
std::shared_ptr<cnn::Tensor3D> cnn::Tensor3D::rot180() const {
    std::shared_ptr<Tensor3D> rot{std::make_shared<Tensor3D>(this->channels_, this->height_, this->width_, "_rot180_")};

    for (int c = 0; c < channels_; ++c) {
        dataType *src = this->data_ + length_ * c;
        dataType *dst = rot->data_ + length_ * c;

        for (int i = 0; i < length_; ++i) {
            dst[i] = src[length_ - 1 - i];
        }
    }

    return rot;
}

/**
 * @brief 对图像数据进行 padding
 * @param padding 填充的size
 * @return 填充之后的 Tensor
 */
std::shared_ptr<cnn::Tensor3D> cnn::Tensor3D::padding(const int padding) const {
    std::shared_ptr<Tensor3D> pad{
            std::make_shared<Tensor3D>(this->channels_, this->height_ + 2 * padding, this->width_ + 2 * padding,
                                       "_padding_" + std::to_string(padding))};
    const uint32_t paddedWidth = this->width_ + 2 * padding;
    const uint32_t paddedSize = (this->height_ + 2 * padding) * paddedWidth;

    for (int c = 0; c < channels_; ++c) {
        for (int i = 0; i < height_; ++i) {
            ::memcpy(pad->data_ + c * paddedSize + (padding + i) * paddedWidth + padding,
                     this->data_ + c * length_ + i * this->width_,
                     sizeof(cnn::dataType) * this->width_);
        }
    }

    return pad;

}

cnn::Tensor3D::~Tensor3D() {
    if (this->data_ != nullptr) {
        delete this->data_;
        this->data_ = nullptr;
    }
}

