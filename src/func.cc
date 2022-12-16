
#include <architectures.hpp>
#include<func.hpp>

inline cnn::dataType __exp(const cnn::dataType x) {
    if (x >= 88) {
        return std::numeric_limits<float>::infinity();
    } else if (x <= -50) {
        return 0.f;
    }

    return std::exp(x);
}

std::vector<cnn::tensor> softMax(const std::vector<cnn::tensor> &input) {
    const int batchSize = input.size();
    const int numOfClasses = input.front()->Length();
    std::vector<cnn::tensor> output;
    output.reserve(batchSize);

    for (int b = 0; b < batchSize; ++b) {
        cnn::tensor probs = std::make_shared<cnn::Tensor3D>(numOfClasses);
        // 首先计算输出的最大值，防止溢出
        const cnn::dataType maxValue = input.at(b)->max();
        cnn::dataType sumValue = 0;
        for (int i = 0; i < numOfClasses; ++i) {
            probs->getData()[i] = __exp(input.at(b)->getData()[i] - maxValue);
            sumValue += probs->getData()[i];
        }

        for (int i = 0; i < numOfClasses; ++i) {
            probs->getData()[i] /= sumValue;
            if (std::isnan(probs->getData()[i])) {
                probs->getData()[i] = 0.f;
            }
        }
        output.emplace_back(std::move(probs));
    }

    return output;
}

std::vector<cnn::tensor> oneHot(const std::vector<int> &labels, const int numOfClasses) {
    const int batchSize = labels.size();
    std::vector<cnn::tensor> oneHotCode;
    oneHotCode.reserve(batchSize);

    for (int b = 0; b < batchSize; ++b) {
        cnn::tensor sample = std::make_shared<cnn::Tensor3D>(numOfClasses);
        sample->setZero();

        assert(labels.at(b) >= 0 && labels.at(b) < numOfClasses);
        sample->getData()[labels[b]] = 1.0;
        oneHotCode.emplace_back(sample);
    }

    return oneHotCode;
}

std::pair<cnn::dataType, std::vector<cnn::tensor>>
crossEntropyBackward(const std::vector<cnn::tensor> &probs, const std::vector<cnn::tensor> &labels) {
    //最小化KL散度等同于最小化交叉熵。
    const int batchSize = labels.size();
    const int numOfClasses = probs.front()->Length();

    std::vector<cnn::tensor> delta;

    delta.reserve(batchSize); //预分配内存

    cnn::dataType lossValue = 0;
    for (int b = 0; b < batchSize; ++b) {
        cnn::tensor piece = std::make_shared<cnn::Tensor3D>(numOfClasses);
        for (int i = 0; i < numOfClasses; ++i) {
            piece->getData()[i] = probs.at(b)->getData()[i] - labels.at(b)->getData()[i];
            lossValue += std::log(probs.at(b)->getData()[i]) * labels.at(b)->getData()[i];
        }
        delta.emplace_back(piece);
    }

    lossValue = lossValue * (-1.0) / batchSize;
    return {lossValue, delta};
}

std::string floatToString(const float value, const int precision) {
    std::stringstream buffer;
    buffer.precision(precision);
    buffer.setf(std::ios::fixed);
    buffer << value;

    return buffer.str();
}
