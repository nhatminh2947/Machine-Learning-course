//
// Created by nhatminh2947 on 11/18/19.
//

#include <matrix.h>
#include "EMAlgorithm.h"

EMAlgorithm::EMAlgorithm(int n_class) : n_class_(n_class), probability_(Col<double>(n_class)),
                                        weight_(Col<double>(n_class)) {
    res_ones = new double[n_class];
    res_zeros = new double[n_class];
}

void EMAlgorithm::Expectation(std::vector<Matrix<double>> image) {
    for (int i = 0; i < n_class_; ++i) {
        res_ones[i] = (weight_ * probability_) * (1 / (weight_[i] * probability_[i]));
        res_zeros[i] = weight_[i] * (1 - probability_[i]) / (weight_ * (1 - probability_));
    }
}

void EMAlgorithm::Maximization(Matrix<double> image) {
    int ones = CountOnes(image);
    int zeros = CountZeros(image);

    for (int i = 0; i < n_class_; ++i) {
        weight_[i] = (ones * res_ones[i] + zeros * res_zeros[i]) / (ones + zeros);
        probability_[i] = (ones * res_ones[i]) / (ones * res_ones[i] + zeros * res_zeros[i]);
    }
}

int EMAlgorithm::CountOnes(Matrix<double> image) {
    int ones = 0;
    for (int i = 0; i < image.getRows(); ++i) {
        for (int j = 0; j < image.getCols(); ++j) {
            ones += (image(i, j) == 1);
        }
    }
    return ones;
}

int EMAlgorithm::CountZeros(Matrix<double> image) {
    int zeros = 0;
    for (int i = 0; i < image.getRows(); ++i) {
        for (int j = 0; j < image.getCols(); ++j) {
            zeros += (image(i, j) == 0);
        }
    }
    return zeros;
}
