//
// Created by nhatminh2947 on 10/10/19.
//

#include "GaussianNaiveBayes.h"

void GaussianNaiveBayes::fit(std::vector<Matrix> X, std::vector<int> y) {
    _train_size = X.size();
    Matrix mu[10];
    Matrix sigma_square[10];
    Matrix x_square[10];

    for (int id = 0; id < _train_size; ++id) {
        _prior(y[id], 0)++;
    }

    for (int i = 0; i < N_CLASSES; ++i) {
        mu[i] = sigma_square[i] = x_square[i] = Matrix(28, 28);
    }

    for (int i = 0; i < _train_size; ++i) {
        mu[y[i]] = mu[y[i]] + X[i];
        x_square[y[i]] = x_square[y[i]] + X[i].power(2);
    }

    for (int i = 0; i < N_CLASSES; ++i) {
        mu[i] = mu[i] / _prior(i, 0);

        _mean[i] = mu[i].flat();

        x_square[i] = x_square[i] / _prior(i, 0);
        x_square[i] = x_square[i].flat();

        for (int row_id = 0; row_id < N_ROWS; ++row_id) {
            for (int col_id = 0; col_id < N_COLS; ++col_id) {
                _variance[i](0, row_id * 28 + col_id) = x_square[i](0, row_id * 28 + col_id) -
                                                        pow(_mean[i](0, row_id * 28 + col_id), 2);

                if (_variance[i](0, row_id * 28 + col_id) == 0) {
                    _variance[i](0, row_id * 28 + col_id) = 1500; // SMOOTHING VALUE
                }
            }
        }
    }

    _prior = _prior / _train_size;
}

int GaussianNaiveBayes::predict(Matrix x) {
    int prediction = -1;
    double min_prob = INT64_MAX;

    std::vector<double> log_proba = predict_log_proba(x, true);

    for (int class_id = 0; class_id < N_CLASSES; ++class_id) {
        std::cout << class_id << ": " << log_proba[class_id] << std::endl;

        if (min_prob > log_proba[class_id]) {
            min_prob = log_proba[class_id];
            prediction = class_id;
        }
    }

    return prediction;
}

std::vector<double> GaussianNaiveBayes::predict_log_proba(Matrix x, bool norm) {
    std::vector<double> result;
    result.reserve(N_CLASSES);
    x = x.flat();

    for (int i = 0; i < N_CLASSES; ++i) {
        double probability = 0.0;
        for (int j = 0; j < N_ROWS * N_COLS; ++j) {
//			double v = (_variance[i](0, j) == 0) ? 100 : _variance[i](0, j);
            probability += log(1.0 / (sqrt(_variance[i](0, j) * 2 * M_PI))) -
                           (0.5 * pow((x(0, j) - _mean[i](0, j)), 2) / _variance[i](0, j));
        }
        probability += log(_prior(i, 0));

        result.push_back(probability);
    }

    if (norm) result = normalize(result);

    return result;
}

GaussianNaiveBayes::GaussianNaiveBayes() {
    for (int i = 0; i < N_CLASSES; ++i) {
        _mean[i] = _variance[i] = Matrix(1, 28 * 28);
    }
}

void GaussianNaiveBayes::imagination() {
    for (int class_id = 0; class_id < N_CLASSES; ++class_id) {
        std::cout << class_id << ":" << std::endl;
        for (int row_id = 0; row_id < N_ROWS; ++row_id) {
            for (int col_id = 0; col_id < N_COLS; ++col_id) {
                int feature_id = 28 * row_id + col_id;

                std::cout << int(_mean[class_id](0, feature_id) >= 128) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}