#pragma once

#include<iostream>
#include<vector>
#include<opencv2/core.hpp>


namespace cnn {
    // tensor 数据类型
    using dataType = float;


    class Tensor3D {
    private:
        const uint32_t channels_;
        const uint32_t height_;
        const uint32_t width_;
        const uint32_t length_; // channels_*height_width_

        dataType *data_;

        std::string name_;
    public:
        const uint32_t getChannels() const {
            return channels_;
        }

        const uint32_t getHeight() const {
            return height_;
        }

        const uint32_t getWidth() const {
            return width_;
        }

        dataType *getData() const {
            return data_;
        }

        Tensor3D(uint32_t channel, uint32_t height, uint32_t width, std::string name = {"pipeline"}) :
                channels_(channel), height_(height), width_(width), name_(std::move(name)),
                length_(height_ * width_ * channels_) {
            this->data_ = new dataType[channel * height * width];
        }

        Tensor3D(std::tuple<uint32_t, uint32_t, uint32_t> &shape, std::string name = {"pipeline"}) :
                channels_(std::get<0>(shape)),
                height_(std::get<1>(shape)),
                width_(std::get<2>(shape)),
                name_(std::move(name)), length_(height_ * width_ * channels_) {
            this->data_ = new dataType[std::get<0>(shape) * std::get<1>(shape) * std::get<2>(shape)];
        }


        Tensor3D(const int length, std::string name = {"pipeline"}) : channels_(length), height_(1), width_(1),
                                                                      length_(height_ * width_ * channels_) {
            this->data_ = new dataType[length];
        }


        void readData(cv::Mat &image);

        void readData(const uchar *image, uint32_t size);

        void setZero();

        dataType max() const;

        uint32_t argmax() const;

        dataType min() const;

        uint32_t argmin() const;


        void div(const dataType times);

        void normalize(const std::vector<dataType> &mean = {0.406, 0.456, 0.485}, const std::vector<dataType> &
        standardDeviation = {0.225, 0.224, 0.229});

        cv::Mat opencvMat(const int ch = 3) const;

        uint32_t length() const;

        std::tuple<uint32_t, uint32_t, uint32_t> shape() const;

        void printShape() const;

        void print(uint32_t channel = 0) const;

        std::shared_ptr<Tensor3D> rot180() const;

        std::shared_ptr<Tensor3D> padding(const int padding) const;

        ~Tensor3D();
    };

    using tensor = std::shared_ptr<Tensor3D>;
}