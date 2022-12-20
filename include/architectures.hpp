#pragma once

#include<fstream>
#include<list>
#include<pipeline.hpp>


namespace cnn::architectures {
    // 随机初始化使用
    extern dataType randomTimes;

    // 是否要backward
    extern bool noGrad;

    //使用 RAII 原则
    class WithOutGrad {
    public:
        explicit WithOutGrad() {
            noGrad = true;
        }

        ~WithOutGrad() noexcept {
            noGrad = false;
        }
    };

    class Layer {
    public:
        std::string name_; //当前层的张量
        std::vector<tensor> output_;// 当前层输出的张量

    public:
        Layer(std::string name) : name_(std::move(name)) {};

        virtual std::vector<tensor> forward(const std::vector<tensor> &input) = 0;

        virtual std::vector<tensor> backward(std::vector<tensor> &delta) = 0;

        virtual void updateGradients(const dataType learningRate = 1e-4) {};

        virtual void saveWeights(std::ofstream &writer) {};

        virtual void loadWeights(std::ifstream &reader) {};

        virtual std::vector<tensor> getOutput() {
            return this->output_;
        }
    };

    class Conv2D : public Layer {
        // 卷积层的固有信息
        std::vector<tensor> weights_; // 权重
        std::vector<dataType> bias_; //偏置

        const int outChannels_;
        const int inChannels_;

        const int kernelSize_; // 卷积核大小
        const int stride_; // 卷积步长

        const int paramsForAKernel_;

        const int padding_; // 填充的大小

        std::default_random_engine seed_;
        std::vector<int> offset_; //执行卷积操作时辅助使用

        std::vector<tensor> _input_; //求梯度需要，即反向传播过程

        // 缓冲区
        std::vector<tensor> deltaOutput_;   //反向传播时传给上一层的梯度
        std::vector<tensor> weightsGradients_;  // 权重的梯度
        std::vector<dataType> biasGradients_;        // bias 的梯度

    public:
        Conv2D(const std::string &name, const int inChannels = 3, const int outChannels = 16, const int kernelSize = 3,
               const int stride = 2) : Layer(name), outChannels_(outChannels), inChannels_(inChannels),
                                       kernelSize_(kernelSize), stride_(stride), padding_(0),
                                       paramsForAKernel_(inChannels_ * kernelSize_ * kernelSize_),
                                       bias_(outChannels), offset_(kernelSize_ * kernelSize_) {
            assert(kernelSize_ & 1 && kernelSize_ >= 3);
            assert(inChannels_ > 0 && outChannels_ > 0 && stride_ > 0);

            weights_.reserve(outChannels_);
            this->seed_.seed(212);
            std::normal_distribution<float> engine(0.0, 1.0);
            for (int i = 0; i < outChannels_; ++i) {
                // 创建 weight 的实体并初始化
                weights_.emplace_back(
                        std::make_shared<Tensor3D>(inChannels_, kernelSize_, kernelSize_,
                                                   this->name_ + "_" + std::to_string(i)));
                bias_.at(i) = engine(this->seed_) / randomTimes;  // bias 偏置初始化
                dataType *dataPtr = this->weights_.at(i)->getData(); // 卷积核权重初始化
                for (int k = 0; k < paramsForAKernel_; ++k) {
                    float random = engine(this->seed_);
                    dataPtr[k] = -random / randomTimes;
//                    printf("conv2d constructor data_ptr[%d]== %lf  random=%lf  random/randomTimes%lf\n", k, dataPtr[k],
//                           random,
//                           random / randomTimes);
                }
            }
        }

        int getParamsNum() const;

        std::vector<tensor> forward(const std::vector<tensor> &input) override;

        std::vector<tensor> backward(std::vector<tensor> &delta) override;

        void updateGradients(dataType learningRate) override;

        void saveWeights(std::ofstream &writer) override;

        void loadWeights(std::ifstream &reader) override;

    private:

        void initForward(int batchSize, std::tuple<uint32_t, uint32_t, uint32_t> &shape, int preWeight);

        void initBackward(std::vector<tensor> &v, int size,
                          std::tuple<uint32_t, uint32_t, uint32_t> shape, std::string name);

        void
        calWeightAndBiasGradients(std::vector<tensor> &delta, uint32_t outHeight, uint32_t outWidth, uint32_t height,
                                  uint32_t width);

        void
        calDeltaGradients(std::vector<tensor> &delta, uint32_t height, uint32_t width, uint32_t inHeight,
                          uint32_t inWidth);
    };


    class MaxPool2D : public Layer {
    private:
        const int kernelSize_; //核大小
        const int step_; // 步长
        const int padding_;// 暂不支持

        // 缓冲区
        std::vector<std::vector<int>> mask_;
        std::vector<tensor> deltaOutput_;
        std::vector<int> offset_;

    public:
        MaxPool2D(const std::string &name, const int kernelSize = 2, const int step = 2) :
                Layer(name),
                kernelSize_(kernelSize),
                step_(step),
                padding_(0),
                offset_(kernelSize * kernelSize) {};

        std::vector<tensor> forward(const std::vector<tensor> &input) override;

        std::vector<tensor> backward(std::vector<tensor> &delta) override;

    private:
        void init(int batchSize, std::tuple<uint32_t, uint32_t, uint32_t> shape, uint32_t height, uint32_t width);
    };

    class ReLU : public Layer {
    public:
        explicit ReLU(const std::string &name) : Layer(name) {}

        std::vector<tensor> forward(const std::vector<tensor> &input) override;

        std::vector<tensor> backward(std::vector<tensor> &delta) override;

    private:
        void init(int size, std::tuple<uint32_t, uint32_t, uint32_t> shape);
    };

    class LinearLayer : public Layer {

    public:
        const int inChannels_; //输入的神经元的个数
        const int outChannels_;//输出的神经元的个数

        std::vector<dataType> weights_; // 权重
        std::vector<dataType> bias_; // 偏置

        //历史信息
        std::tuple<uint32_t, uint32_t, uint32_t> deltaShape_;
        std::vector<tensor> _input_;

        // 缓冲区
        std::vector<tensor> deltaOutPut_;
        std::vector<dataType> weightGradients_;
        std::vector<dataType> biasGradients_;

    public:
        LinearLayer(const std::string &name, const int inChannels, const int outChannels) :
                Layer(name),
                inChannels_(inChannels),
                outChannels_(outChannels), weights_(inChannels_ * outChannels_, 0),
                bias_(outChannels_, 0) {

            std::default_random_engine e(1899);
            std::normal_distribution<float> engine(0.0, 1.0);
            //bias 随机出初始化
            for (int i = 0; i < outChannels_; ++i) {
                bias_.at(i) = engine(e) / randomTimes;
                //std::cout << "bias_ linear constructor" << bias_.at(i) << std::endl;
            }
            // weight_ 随机初始化

            int length = inChannels_ * outChannels_;
            for (int i = 0; i < length; ++i) {
                weights_.at(i) = engine(e) / randomTimes;
            }
        }

        std::vector<tensor> forward(const std::vector<tensor> &input) override;

        std::vector<tensor> backward(std::vector<tensor> &delta) override;

        virtual void updateGradients(const dataType learningRate = 1e-4) override;

        virtual void saveWeights(std::ofstream &writer) override;

        virtual void loadWeights(std::ifstream &reader) override;

        void calWeightGradients(std::vector<tensor> &delta);

        void calBiasGradients(std::vector<tensor> &delta);

        void calInputGradients(std::vector<tensor> &delta);
    };


    class BatchNorm2D : public Layer {
    private:
        // 固有信息
        const int outChannels_;
        const dataType eps_;
        const dataType momentNum_;

        //要进行学习的参数
        std::vector<dataType> gamma_;
        std::vector<dataType> beta_;

        // 要保留的历史信息
        std::vector<dataType> movingMean_;
        std::vector<dataType> movingVar_;

        // 缓冲区
        std::vector<tensor> normedInput_;
        std::vector<dataType> bufferMean_;
        std::vector<dataType> bufferVar_;

        // 保留的梯度信息
        std::vector<dataType> gammaGradients_;
        std::vector<dataType> betaGradients_;

        // 临时的梯度信息
        tensor normGradients_;

        // 求梯度需要
        std::vector<tensor> _input_;

    public:
        BatchNorm2D(const std::string &name, const int outChannels, const dataType eps = 1e-4,
                    const dataType momentNum = 0.1) :
                Layer(name),
                outChannels_(outChannels),
                eps_(eps),
                momentNum_(momentNum),
                gamma_(outChannels, 0),
                beta_(outChannels, 0),
                movingMean_(outChannels, 0),
                movingVar_(outChannels, 0),
                bufferMean_(outChannels, 0),
                bufferVar_(outChannels, 0) {}

        std::vector<tensor> forward(const std::vector<tensor> &input) override;

        std::vector<tensor> backward(std::vector<tensor> &delta) override;

        void updateGradients(const dataType learningRate) override;

        void saveWeights(std::ofstream &writer) override;

        void loadWeights(std::ifstream &reader) override;

        std::vector<tensor> getOutput() override;

    private:
        void init(std::tuple<uint32_t, uint32_t, uint32_t> &&shape);

        void forwardWithOutGrad();

        void forwardWithGrad();
    };

    class AlexNet {
    public:
        bool printInfo = false;
    private:
        std::list<std::shared_ptr<Layer>> layerSequence_;

    public:
        AlexNet(const int numOfClasses = 3, const bool batchNorm = false);

        std::vector<tensor> forward(const std::vector<tensor> &input);

        void backward(std::vector<tensor> &delta);

        void updateGradients(const dataType learningRate = 1e-4);

        void saveWeights(const std::filesystem::path &path) const;

        void loadWeights(const std::filesystem::path &path);

        cv::Mat gradCam(const std::string &layerName) const;
    };


    void printTensor(const std::vector<cnn::tensor> &input);
}