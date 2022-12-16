#pragma once

#include<data_format.hpp>

std::vector<cnn::tensor> softMax(const std::vector<cnn::tensor> &input);

std::vector<cnn::tensor> oneHot(const std::vector<int> &labels, const int numOfClasses);

std::pair<cnn::dataType, std::vector<cnn::tensor>> crossEntropyBackward(
        const std::vector<cnn::tensor> &probs, const std::vector<cnn::tensor> &labels
);

std::string floatToString(const float value, const int precision);

