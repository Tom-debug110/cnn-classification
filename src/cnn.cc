#include<memory>
#include<iostream>
#include<filesystem>

#include<architectures.hpp>
#include<metrics.hpp>
#include<func.hpp>
#include<utility>
// hello
int main(int argc, char **argv) {

    setbuf(stdout, 0);

    std::cout << "OpenCV " << CV_VERSION << std::endl;
    std::cout << "Clang " << __VERSION__ << std::endl;

    const int trainBatchSize = 4;
    const int validBatchSize = 1;
    const int testBatchSize = 1;

    assert(trainBatchSize >= validBatchSize && validBatchSize >= testBatchSize);
    assert(validBatchSize == 1 && testBatchSize == 1);

    const std::tuple<uint32_t, uint32_t, uint32_t> imageSize{224, 224, 3};

    const std::filesystem::path datasetPath{"../datasets/animals"};
    const std::vector<std::string> categories{"dog", "panda", "bird"};

    auto dataset = cnn::pipeline::getImagesForClassification(datasetPath, categories);

    // 构造数据流
    cnn::pipeline::DataLoader trainLoader(dataset["train"], trainBatchSize, false, true, imageSize);
    cnn::pipeline::DataLoader validLoader(dataset["valid"], validBatchSize, false, false, imageSize);

    // 定义网络结构
    const int numOfClasses = categories.size();
    cnn::architectures::AlexNet alexNet(numOfClasses, false);

    const std::filesystem::path checkPointDir{"./check_points/AlexNet_aug_1e-3"};
    if (not std::filesystem::exists(checkPointDir))
        std::filesystem::create_directories(checkPointDir);
    std::filesystem::path best_checkpoint;  // 当前正确率最高的模型
    float currentBestAccuracy = -1; // 记录当前最高的正确率

    const int startIters = 1; //从第几个 iter 开始
    const int totalIters = 40000; //迭代次数
    const float learningRate = 1e-3;// 学习率

    const int validInters = 1000;// 验证一次的间隔
    const int saveIters = 5000;// 保存一次的间隔
    float meanLoss = 0;//平均损失
    float curIter = 0;//计算平均损失

    ClassificationEvaluator trainEvaluator; //计算累积的准确率
    std::vector<int> predict(trainBatchSize, -1);//  存储每个 batch 的预测结果 计算准确率使用


    for (int i = startIters; i < totalIters; ++i) {

        const auto sample = trainLoader.generateBatch();

        const auto out = alexNet.forward(sample.first);

        const auto probs = softMax(out);

        auto lossDelta = crossEntropyBackward(probs, oneHot(sample.second, numOfClasses));

        meanLoss += lossDelta.first;

        alexNet.backward(lossDelta.second);

        alexNet.updateGradients(learningRate);

        for (int b = 0; b < trainBatchSize; ++b) {
            predict[b] = probs[b]->argmax();
        }

        trainEvaluator.compute(predict, sample.second);
        ++curIter;

        printf("\rTrain===> [batch %d/%d] [loss %.3f] [Accuracy %4.3f]", i, totalIters, meanLoss / curIter,
               trainEvaluator.get());


        // 开始验证
        if (i % validInters == 0) {
            printf("\n[开始验证]\n\n");
            cnn::architectures::WithOutGrad guard;
            float meanValidLoss = 0.f;
            ClassificationEvaluator validEvaluator;

            const int samplesNum = validLoader.length();

            for (int s = 0; s < samplesNum; ++s) {
                const auto validSample = validLoader.generateBatch();
                const auto validOut = alexNet.forward(sample.first);
                const auto _probs = softMax(validOut);

                const auto validLossDelta = crossEntropyBackward(_probs, oneHot(sample.second, numOfClasses));

                meanValidLoss += validLossDelta.first;
                for (int j = 0; j < trainBatchSize; ++j) {
                    predict[j] = _probs[j]->argmax();
                }

                validEvaluator.compute(predict, sample.second);

                printf("\rValid===> [batch %d/%d] [loss %.3f] [Accuracy %4.3f]", s, samplesNum, meanValidLoss / s,
                       validEvaluator.get());
            }

            printf("\n\n");

            if (i % saveIters == 0) {
                const float trainAccuracy = trainEvaluator.get();
                const float validAccuracy = validEvaluator.get();

                // 决定保存的名字
                std::string save_string("iter_" + std::to_string(i));
                save_string += "_train_" + floatToString(trainAccuracy, 3);
                save_string += "_valid_" + floatToString(validAccuracy, 3) + ".model";
                std::filesystem::path save_path = checkPointDir / save_string;
                // 保存权值
                alexNet.saveWeights(save_path);
                // 记录最佳的正确率和对应的路径
                if (validAccuracy > currentBestAccuracy) {
                    best_checkpoint = save_path;
                    currentBestAccuracy = validAccuracy;
                }
            }

            curIter = 0;
            meanLoss = 0;
            trainEvaluator.clear();
        }
    }
    return 0;
}