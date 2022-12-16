#include<architectures.hpp>
#include<pipeline.hpp>
#include<random>
#include<vector>
#include<opencv2/highgui.hpp>


void augmentTest() {
    cnn::pipeline::ImageAugmentor augmentor({{"hflip",  1.0},
                                             {"vflip",  0.1},
                                             {"crop",   1.0},
                                             {"rotate", 1.0}});
    cv::Mat origin = cv::imread("../datasets/images/dog.jpg");

    assert(!origin.empty());

    cnn::pipeline::display(origin, "ok");
    augmentor.makeAugment(origin, true);

    origin = cv::imread("../datasets/images/bird.jpg");
    augmentor.makeAugment(origin, true);

    origin = cv::imread("../datasets/images/cat.jpg");
    augmentor.makeAugment(origin, true);

    origin = cv::imread("../datasets/images/yuan.jpg");
    augmentor.makeAugment(origin, true);

    origin = cv::imread("../datasets/images/panda.jpg");
    augmentor.makeAugment(origin, true);

    origin = cv::imread("../datasets/images/bird_3.jpg");
    augmentor.makeAugment(origin, true);
}

void dataLoaderTest() {
    std::cout << "OpenCV " << CV_VERSION << std::endl;
    // 指定一些参数
    const int trainBatchSize = 4;
    const std::tuple<uint32_t, uint32_t, uint32_t> imageSize({224, 224, 3});
    const std::filesystem::path datasetPath("../datasets/animals");
    const std::vector<std::string> categories({"dog", "panda", "bird"});

    auto dataset = cnn::pipeline::getImagesForClassification(datasetPath, categories);

    cnn::pipeline::DataLoader trainLoader(dataset["train"], trainBatchSize, false, true, imageSize);

    for (int i = 0; i < 10; ++i) {
        auto sample = trainLoader.generateBatch();
        const auto &images = sample.first;
        const auto &labels = sample.second;

        for (int b = 0; b < trainBatchSize; ++b) {
            std::cout << "[Batch " << i << "] " << " [" << b + 1 << "/" << trainBatchSize << "]===> "
                      << categories[labels[b]] << std::endl;

            const auto origin = images[b]->opencvMat(3);
            cnn::pipeline::display(origin, "ok" + std::to_string(b));
        }
    }
}


void tensorTest() {
    cnn::Tensor3D t(3, 5, 5);
    cv::Mat original = cv::imread("../datasets/images/dog.jpg");
    //std::cout << original.size << std::endl;

    cv::Mat test = cv::Mat::ones({5, 5}, CV_8UC3);
    //std::cout << test << std::endl;
    t.readData(test);
    cv::Mat dd = t.opencvMat(3);

    cnn::pipeline::display(test, "ok");

}


void ReLUTest() {
    std::vector<cnn::tensor> input;
    std::tuple<uint32_t, uint32_t, uint32_t> shape{16, 7, 7};

    input.emplace_back(new cnn::Tensor3D(shape));

    //随机数生成引擎
    std::default_random_engine e;
    e.seed(std::chrono::steady_clock::now().time_since_epoch().count());


    std::normal_distribution<float> engine(0.0, 1.0);

    // 初始化
    cnn::dataType *dataPtr = input.front()->getData();
    const int length = input.front()->Length();

    for (int i = 0; i < length; ++i) {
        dataPtr[i] = engine(e);
    }

    // 打印第 0 张图像特征的第三个通道
    input.front()->print(1);

    // 声明ReLU 层

    cnn::architectures::ReLU reLu("relu_test");
    auto out = reLu.forward(input);
    out.front()->print(1);

    // 模拟反向传播回来的 delta
    std::vector<cnn::tensor> delta({cnn::tensor(std::make_shared<cnn::Tensor3D>(shape))});
    for (int i = 0; i < length; ++i) {
        delta.front()->getData()[i] = engine(e);
    }

    delta.front()->print(1);

    // 计算反向传播
    auto deltaBackward = reLu.backward(delta);

    deltaBackward.front()->print(1);
}


void maxPool2DTest() {
    std::vector<cnn::tensor> input;
    std::tuple<uint32_t, uint32_t, uint32_t> shape{1, 6, 6};

    input.emplace_back(std::make_shared<cnn::Tensor3D>(shape));

    std::default_random_engine e;
    e.seed(std::chrono::steady_clock::now().time_since_epoch().count());

    std::normal_distribution<float> engine(0, 1);

    // 初始化
    cnn::dataType *dataPtr = input.front()->getData();
    int length = input.front()->Length();

    for (int i = 0; i < length; ++i) {
        dataPtr[i] = engine(e);
    }

    const int channel = 1;
    input.front()->print(channel);

    //声明 MaxPool 层
    cnn::architectures::MaxPool2D maxPool2D("maxPoolTest", 2, 2);

    auto out = maxPool2D.forward(input);

    out.front()->print(channel);
    out.front()->printShape();

    // 模拟反向传播也就是下一层传回来的梯度
    shape = {1, 3, 3};
    std::vector<cnn::tensor> delta({std::make_shared<cnn::Tensor3D>(shape)});

    length = delta.front()->Length();
    for (int i = 0; i < length; ++i) {
        delta.front()->getData()[i] = engine(e);
    }

    delta.front()->print(channel);

    auto deltaBackward = maxPool2D.backward(delta);
    deltaBackward.front()->print(channel);

}

void Conv2DTest() {
    std::vector<cnn::tensor> input;
    std::tuple<uint32_t, uint32_t, uint32_t> shape{3, 224, 224};

    input.emplace_back(std::make_shared<cnn::Tensor3D>(shape));

    std::default_random_engine e;
    e.seed(std::chrono::steady_clock::now().time_since_epoch().count());

    std::normal_distribution<float> engine(0, 1);

    // 初始化
    cnn::dataType *dataPtr = input.front()->getData();
    int length = input.front()->Length();

    for (int i = 0; i < length; ++i) {
        dataPtr[i] = engine(e);
    }

    const int channel = 1;
    //input.front()->print(channel);

    cnn::architectures::Conv2D conv2D("conv_test", 16, 3, 3, 3);
    auto out = conv2D.forward(input);

    //out.front()->print(1);
    out.front()->printShape();


}

void AlexNetTest() {
    cnn::architectures::AlexNet alexNet(3, false);
    alexNet.printInfo = true;

    std::vector<cnn::tensor> input;

    const int batchSize = 1;

    for (int i = 0; i < batchSize; ++i) {
        input.emplace_back(std::make_shared<cnn::Tensor3D>(3, 9, 9));
    }

    std::default_random_engine e;
    e.seed(std::chrono::steady_clock::now().time_since_epoch().count());

    std::uniform_int_distribution<int> engine(0, 200);

    // 初始化
    cnn::dataType *dataPtr = input.front()->getData();
    int length = input.front()->Length();

    for (int i = 0; i < length; ++i) {
        dataPtr[i] = engine(e);
    }

    auto out = alexNet.forward(input);

    cnn::architectures::printTensor(out);


    std::vector<cnn::tensor> delta;

    for (int i = 0; i < batchSize; ++i) {
        delta.emplace_back(std::make_shared<cnn::Tensor3D>(3, 1, 1, "delta_from_loss" + std::to_string(i)));
    }

    alexNet.backward(delta);
}

int main1(int argc, char **argv) {

//    augmentTest();
//
//    dataLoaderTest();
//
//    tensorTest();
//
//    ReLUTest();
//
//    maxPool2DTest();

//    Conv2DTest();

    AlexNetTest();
    return 0;

}