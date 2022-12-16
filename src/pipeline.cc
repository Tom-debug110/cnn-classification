
#include <pipeline.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include <utility>


void cnn::pipeline::display(cv::Mat image, std::string win) {
    cv::imshow(win, image);
    cv::waitKey(10);
    cv::destroyAllWindows();
}

bool cnn::pipeline::writeByOpenCV(const cv::Mat &source, const std::string path) {
    return cv::imwrite(path + ".jpg", source, std::vector<int>{cv::IMWRITE_PNG_COMPRESSION, 0});
}

cv::Mat cnn::pipeline::rotate(cv::Mat &src, float angle) {
    cv::Point2f center((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
    rot.at<double>(0, 2) += bbox.width / 2.0 - src.cols / 2.0;
    rot.at<double>(1, 2) += bbox.height / 2.0 - src.rows / 2.0;
    cv::warpAffine(src, src, rot, bbox.size());
    return src;
}

int cnn::pipeline::DataLoader::length() const {
    return this->imageNum_;
}

std::pair<std::vector<cnn::tensor>, std::vector<int>> cnn::pipeline::DataLoader::generateBatch() {
    std::vector<tensor> images;
    std::vector<int> labels;
    images.reserve(this->batchSize_);
    labels.reserve(this->batchSize_);

    for (int i = 0; i < batchSize_; ++i) {
        auto sample = this->addToBuffer_(i);
        images.emplace_back(sample.first);
        labels.emplace_back(sample.second);
    }

    return std::make_pair(std::move(images), std::move(labels));
}

std::pair<cnn::tensor, int> cnn::pipeline::DataLoader::addToBuffer_(const int batchIndex) {
    // 获取图像的序号
    ++this->iterator_;
    if (this->iterator_ == this->imageNum_) {
        this->iterator_ = 0;
        if (this->shuffle_) {
            std::shuffle(images_.begin(), images_.end(), std::default_random_engine(789));
        }
    }

    // 读取图像
    auto image = this->images_.at(iterator_);
    std::string path = image.first;
    int label = image.second;

    cv::Mat origin = cv::imread(path);
    if (this->augment_) {
        imageAugmentor_.makeAugment(origin);
    }

    // resize 必须在数据增强之后
    cv::resize(origin, origin, {static_cast<int>(width_), static_cast<int>(height_)});

    //从 OpenCV
    this->buffer_.at(batchIndex)->readData(origin);

    // 返回图像的内容和 buffer
    //this->buffer_.at(batchIndex)->opencvMat(3);
    return {this->buffer_.at(batchIndex), label};
}

cnn::pipeline::DataLoader::DataLoader(cnn::pipeline::listType images, const uint32_t batchSize,
                                      const bool augment, const bool shuffle,
                                      std::tuple<uint32_t, uint32_t, uint32_t> imageSize, const int seed) :
        images_(std::move(images)),
        batchSize_(batchSize),
        augment_(augment),
        shuffle_(shuffle),
        seed_(seed),
        height_(std::get<0>(imageSize)),
        width_(std::get<1>(imageSize)),
        channels_(std::get<2>(imageSize)) {

    this->imageNum_ = this->images_.size();
    this->buffer_.reserve(this->batchSize_);
    for (int i = 0; i < batchSize_; ++i) {
        buffer_.emplace_back(std::make_shared<cnn::Tensor3D>(channels_, height_, width_));
    }
}

void cnn::pipeline::ImageAugmentor::makeAugment(cv::Mat &origin, const bool show) {
    // 随机打乱次序
    std::shuffle(operations_.begin(), operations_.end(), this->l_);

    // 遍历整个操作
    for (const auto &item: operations_) {
        const float probability = engine_(e_);
        // 概率太小，不执行操作
        if (probability < 1.0 - item.second) {
            continue;
        }

        if (item.first == "hflip") {
            cv::flip(origin, origin, 1);
        } else if (item.first == "vflip") {
            cv::flip(origin, origin, 0);
        } else if (item.first == "crop") {
            const int row = origin.rows;
            const int col = origin.cols;

            float ratio = 0.7f + cropEngine_(c_);

            // 计算机裁剪尺寸
            const int rowAfterCrop = row * ratio;
            const int colAfterCrop = col * ratio;

            // 获取随机的裁剪位置
            std::uniform_int_distribution posOfRow(0, row - rowAfterCrop - 10);
            std::uniform_int_distribution posOfCol(0, col - colAfterCrop - 10);

            origin = origin(cv::Rect(posOfCol(c_), posOfRow(c_), colAfterCrop, rowAfterCrop));
        } else if (item.first == "rotate") {
            float angle = rotateEngine_(r_);
            if (minusEngine_(r_) & 1) {
                angle = -angle;
            }
            origin = cnn::pipeline::rotate(origin, angle);
        }

        if (show) {
            pipeline::display(origin, std::to_string(engine_(e_)));
        }
    }
}

std::map<std::string, cnn::pipeline::listType>
cnn::pipeline::getImagesForClassification(const std::filesystem::path &dataset,
                                          const std::vector<std::string> &categories,
                                          const std::pair<float, float> &ratios) {
    //遍历dataset 文件夹下指定的类别
    listType allImageList;
    int kindNum = 0;
    for (const auto &category: categories) {
        const auto dir = dataset / category;
        std::cout << dir.string() << std::endl;
        assert(std::filesystem::exists(dir));

        auto walker = std::filesystem::directory_iterator(dir);

        for (const auto &item: walker) {
            allImageList.emplace_back(item.path().string(), kindNum);
        }

        ++kindNum;
    }

    // 打乱图像列表
    std::shuffle(allImageList.begin(), allImageList.end(), std::default_random_engine(313));

    const int total = allImageList.size();
    const int trainSize = total * ratios.first;
    const int testSize = total * ratios.second;

    std::map<std::string, listType> results;

    results.emplace("train", listType(allImageList.begin(), allImageList.begin() + trainSize));
    results.emplace("test", listType(allImageList.begin() + trainSize, allImageList.begin() + trainSize + testSize));
    results.emplace("valid", listType(allImageList.begin() + trainSize + testSize, allImageList.end()));

    std::cout << "\ntraint : " << results["train"].size() << "\ntest : " << results["test"].size() << "\nvalid : "
              << results["valid"].size() << std::endl;

    return results;
}
