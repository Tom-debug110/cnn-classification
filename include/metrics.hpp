#pragma once

#include<vector>

class ClassificationEvaluator {
private:
    int correctNum = 0;
    int sampleNum = 0;

public:
    ClassificationEvaluator() = default;

    // 一个batch 猜对了几个
    void compute(const std::vector<int> &predict, const std::vector<int> &labels);

    // 查看累积的正确率
    float get() const;

    //重新开始统计
    void clear();
};

