//
// Created by nhatminh2947 on 10/8/19.
//

#include <cmath>
#include "MNISTNaiveBayesClassifier.h"

void MNISTNaiveBayesClassifier::fit(std::vector<Matrix> X, std::vector<int> y) {
    _train_size = X.size();
    for (int i = 0; i < N_CLASSES; ++i) {
        for (int label : y) {
            if (label == i) {
                _count_class(i, 0)++;
            }
        }

        _prop_class(i, 0) = _count_class(i, 0) / _train_size;
    }

    for (int class_id = 0; class_id < N_CLASSES; ++class_id) {
        for (int feature_id = 0; feature_id < N_FEATURES; ++feature_id) {
            int row_id = feature_id / 28;
            int col_id = feature_id % 28;

            for (int id = 0; id < _train_size; ++id) {
                if (y[id] == class_id) {
                    _weights(class_id, feature_id) += X[id](row_id, col_id);
                }
            }
            _count_feature_class(class_id, 0) += _weights(class_id, feature_id);
        }
    }

    std::cout << _weights << std::endl;
}

int MNISTNaiveBayesClassifier::predict(Matrix x) {
    int prediction = -1;
    double min_log_proba = INT64_MAX;
    double max_proba = INT64_MIN;

//    for (int class_id = 0; class_id < N_CLASSES; ++class_id) {
////        double likelihood = 1.0;
//        double log_likelihood = 0;
//        double prior = _prop_class(class_id, 0);
//        for (int feature_id = 0; feature_id < N_FEATURES; ++feature_id) {
////            likelihood += (x(feature_id, 0) * _weights(class_id, feature_id) / _count_feature_class(class_id, 0));
////            likelihood *= (std::pow(
////                    (_weights(class_id, feature_id) + 1) / (_count_feature_class(class_id, 0) + N_FEATURES),
////                    x(feature_id / 28, feature_id % 28)));
//
//            log_likelihood += (x(feature_id / 28, feature_id % 28) *
//                               (log(_weights(class_id, feature_id) + 1)
//                                - log(_count_feature_class(class_id, 0) + N_FEATURES)));
//        }
//
//        double log_posterior = log_likelihood + log(prior);
//        std::cout << class_id << ": " << log_posterior << std::endl;
//
////        if (min_log_proba > log_posterior) {
////            min_log_proba = log_posterior;
////            prediction = class_id;
////        }
//
//        if (max_proba < log_posterior) {
//            max_proba = log_posterior;
//            prediction = class_id;
//        }
//    }

    std::vector<double> log_proba = predict_log_proba(x, true);

    for (int class_id = 0; class_id < N_CLASSES; ++class_id) {
        std::cout << class_id << ": " << log_proba[class_id] << std::endl;

        if (min_log_proba > log_proba[class_id]) {
            min_log_proba = log_proba[class_id];
            prediction = class_id;
        }
    }

    return prediction;
}

std::vector<double> MNISTNaiveBayesClassifier::predict_log_proba(Matrix x, bool norm) {
    std::vector<double> log_proba;

    for (int class_id = 0; class_id < N_CLASSES; ++class_id) {
        double log_likelihood = 0;
        double prior = _prop_class(class_id, 0);
        for (int feature_id = 0; feature_id < N_FEATURES; ++feature_id) {
            log_likelihood += (x(feature_id / 28, feature_id % 28) *
                               (log(_weights(class_id, feature_id) + 1)
                                - log(_count_feature_class(class_id, 0) + N_FEATURES)));
        }

        double log_posterior = log_likelihood + log(prior);

        log_proba.push_back(log_posterior);
    }

    if(norm) log_proba = normalize(log_proba);
    return log_proba;
}

std::vector<double> normalize(std::vector<double> v) {
    double sum = 0;
    for (double i : v) {
        sum += i;
    }

    for (double & i : v) {
        i /= sum;
    }

    return v;
}

MNISTNaiveBayesClassifier::MNISTNaiveBayesClassifier() = default;
