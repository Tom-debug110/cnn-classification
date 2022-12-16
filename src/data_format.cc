#include<data_format.hpp>
#include<iomanip>

void cnn::Tensor3D::readData(cv::Mat &image) {
    uint32_t size = this->width_ * this->height_;
    readData(image.data, size);
}

void cnn::Tensor3D::readData(const uchar *const image, const uint32_t size) {

//    std::cout << size << std::endl;
    // 最初是按照一行一行的进行复制，但是OpenCV里面本来也是线性的，还使用二维的方式反而多次一举啦
    for (int i = 0; i < size; ++i) {
        uint32_t position = 3 * i;
        this->data_[i] = image[position] * 1.0 / 255;
        this->data_[size + i] = image[position + 1] * 1.0 / 255;
        this->data_[size * 2 + i] = image[position + 2] * 1.0 / 255;
//        ::printf("%d %d %d\n", image[position], image[position + 1], image[position + 2]);
    }
}

void cnn::Tensor3D::setZero() {
    ::memset(this->data_, 0, sizeof(cnn::dataType) * this->length_);
}

cnn::dataType cnn::Tensor3D::max() const {
    return this->data_[argmax()];
}

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

void cnn::Tensor3D::div(const cnn::dataType times) {
    for (int i = 0; i < length_; ++i) {
        this->data_[i] /= times;
    }
}

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

uint32_t cnn::Tensor3D::Length() const {
    return this->length_;
}

std::tuple<uint32_t, uint32_t, uint32_t> cnn::Tensor3D::shape() const {
    return {this->channels_, this->height_, this->width_};
}

void cnn::Tensor3D::printShape() const {
    ::printf("[%d,%d,%d]\n", this->channels_, this->height_, this->width_);
}

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

