#include<architectures.hpp>

cnn::dataType cnn::architectures::randomTimes = 10.f;
bool cnn::architectures::noGrad = false;

void cnn::architectures::printTensor(const std::vector<cnn::tensor> &input) {

    for (int i = 0; i < input.size(); ++i) {
        auto cur = input.at(i);
        printf("input[%d] length==%d\n", i, cur->length());
        for (int j = 0; j < cur->length(); ++j) {
            printf("%lf ", cur->getData()[j]);
        }
        printf("\n");
    }

    std::cout << std::endl;

}