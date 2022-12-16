#include<metrics.hpp>

void ClassificationEvaluator::compute(const std::vector<int> &predict, const std::vector<int> &labels) {
    const int batchSize = labels.size();
    for (int b = 0; b < batchSize; ++b) {
        if (predict.at(b) == labels.at(b)) {
            ++this->correctNum;
        }
    }
    this->sampleNum += batchSize;
}

float ClassificationEvaluator::get() const {
    return this->correctNum * 1.0 / this->sampleNum;
}

void ClassificationEvaluator::clear() {
    this->correctNum = this->sampleNum = 0;
}