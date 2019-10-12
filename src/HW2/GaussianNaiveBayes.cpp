//
// Created by nhatminh2947 on 10/10/19.
//

#include "GaussianNaiveBayes.h"

void GaussianNaiveBayes::fit(std::vector<Matrix> X, std::vector<int> y) {
    _train_size = X.size();
    Matrix mu[10];
    Matrix sigma[10];

    for (int i = 0; i < N_CLASSES; ++i) {
        mu[i] = sigma[i] = Matrix(28, 28);
    }

    for (int i = 0; i < _train_size; ++i) {
        mu[y[i]] = mu[y[i]] + X[i];
    }

    for (int i = 0; i < N_CLASSES; ++i) {
        mu[i] = mu[i] / _train_size;
        _mean[i] = mu[i].flat();
        std::cout << _mean[i];
    }

    for (int i = 0; i < _train_size; ++i) {
        Matrix A = X[i] - mu[y[i]];
        sigma[y[i]] = sigma[y[i]] + A.power(2);
    }

    std::cout << "Variance" << std::endl;
    for (int i = 0; i < N_CLASSES; ++i) {
        sigma[i] = sigma[i] / _train_size;
        _variance[i] = sigma[i].flat();
        std::cout << _variance[i];
    }
}

int GaussianNaiveBayes::predict(Matrix x) {
    int prediction = -1;
    double max_prob = INT64_MAX;

    std::vector<double> log_proba = predict_log_proba(x, true);

    for (int class_id = 0; class_id < N_CLASSES; ++class_id) {
        std::cout << class_id << ": " << log_proba[class_id] << std::endl;

        if (max_prob < log_proba[class_id]) {
            max_prob = log_proba[class_id];
            prediction = class_id;
        }
    }

    return prediction;
}

std::vector<double> GaussianNaiveBayes::predict_log_proba(Matrix x, bool normalize) {
    std::vector<double> result;
    result.reserve(N_CLASSES);

    for (int i = 0; i < N_CLASSES; ++i) {
        double probability = 1.0;
        for (int j = 0; j < 784; ++j) {
            probability *= (1.0 / (2.0 * M_PI * _variance[i](0, j))) *
                          exp((x.flat()(0, j) - _mean[i](0, j)) * (x.flat()(0, j) - _mean[i](0, j)) *
                              (1.0 / (2 * _variance[i](0, j))));
        }

        result.push_back(probability);
    }

    return result;
}

GaussianNaiveBayes::GaussianNaiveBayes() {
    for (int i = 0; i < N_CLASSES; ++i) {
        _mean[i] = _variance[i] = Matrix(1, 28 * 28);
    }
}
