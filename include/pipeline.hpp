#pragma once

#include<vector>
#include<data_format.hpp>
#include<random>
#include<map>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<filesystem>

namespace cnn::pipeline {
    //
    using listType = std::vector<std::pair<std::string, int>>;

    std::map<std::string, listType> getImagesForClassification(
            const std::filesystem::path &dataset,
            const std::vector<std::string> &categories = {},
            const std::pair<float, float> &ratios = {0.8, 0.1}
    );

    class ImageAugmentor {
    private:
        //e 用来获得操作 l 用来打乱操作列表 c 用来裁剪需要的概率 r 用来得到旋转的概率
        std::default_random_engine e_, l_, c_, r_;

        std::uniform_real_distribution<float> engine_;
        std::uniform_real_distribution<float> cropEngine_;
        std::uniform_real_distribution<float> rotateEngine_;
        std::uniform_int_distribution<int> minusEngine_;
        std::vector<std::pair<std::string, int>> operations_; // 操作列表合集

    public:
        explicit ImageAugmentor(const std::vector<std::pair<std::string, int>> &operations = {{"hflip",  0.5},
                                                                                              {"vflip",  0.5},
                                                                                              {"crop",   0.7},
                                                                                              {"rotate", 0.5}}) :
                operations_(operations), e_(212), l_(826), c_(230), r_(520),
                engine_(0.0, 1.0), cropEngine_(0.0, 0.25), rotateEngine_(15, 75), minusEngine_(1, 10) {

        }

        void makeAugment(cv::Mat &origin, const bool show = false);
    };


    class DataLoader {
        using batchType = std::pair<std::vector<cnn::tensor>, std::vector<int>>;

    private:
        listType images_;           // 数据集列表
        int imageNum_;                  // 这个子数据集一共有多少张图像和对应的标签
        const uint32_t batchSize_;  // 每次打包几张图像
        const bool augment_;        // 是否要做图像增强
        const bool shuffle_;        // 是否要打乱列表
        const int seed_;            // 每次随机打乱列表的种子
        int iterator_;              // 当前采集到了第 iterator 张图像
        std::vector<tensor> buffer_;// batch 缓冲区，用来从图像生成 tensor 的

        const uint32_t channels_, width_, height_;

    public:
        DataLoader(listType images, const uint32_t batchSize, const bool augment, const bool shuffle,
                   std::tuple<uint32_t, uint32_t, uint32_t> imageSize = {224u, 224u, 3u},
                   const int seed = 212);

        int length() const;

        batchType generateBatch();

    private:
        std::pair<tensor, int> addToBuffer_(const int batchIndex);

        ImageAugmentor imageAugmentor_;
    };

    void display(cv::Mat image, std::string win);

    bool writeByOpenCV(const cv::Mat &source, const std::string path);

    cv::Mat rotate(cv::Mat &src, float angle);
}