//
// Created by nhatminh2947 on 10/8/19.
//

#include <cmath>
#include "MultinominalNaiveBayes.h"

void MultinominalNaiveBayes::fit(std::vector<Matrix> X, std::vector<int> y) {
	_train_size = X.size();
	for (int i = 0; i < N_CLASSES; ++i) {
		for (int label : y) {
			if (label == i) {
				_count_class(i, 0)++;
			}
		}

		_prop_class(i, 0) = _count_class(i, 0) / _train_size;
	}

	for (int row_id = 0; row_id < 28; ++row_id) {
		for (int col_id = 0; col_id < 28; ++col_id) {
			for (int id = 0; id < _train_size; ++id) {
//				std::cout << X[id](row_id, col_id) << std::endl;
				int feature_id = int(X[id](row_id, col_id)) * 28 * 28 + row_id * 28 + col_id;
//				std::cout << "Feature id: " << feature_id << std::endl;
				_weights(y[id], feature_id) += 1;
				_count_feature_class(y[id], 0) += 1;
			}
		}
	}
	std::cout << _weights << std::endl;
}

int MultinominalNaiveBayes::predict(Matrix x) {
	int prediction = -1;
	double min_log_proba = INT64_MAX;

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

std::vector<double> MultinominalNaiveBayes::predict_log_proba(Matrix x, bool norm) {
	std::vector<double> log_proba;

	for (int class_id = 0; class_id < N_CLASSES; ++class_id) {
		double log_likelihood = 0;
		double prior = _prop_class(class_id, 0);

		for (int row_id = 0; row_id < 28; ++row_id) {
			for (int col_id = 0; col_id < 28; ++col_id) {

				int feature_id = int(x(row_id, col_id)) * 28 * 28 + row_id * 28 + col_id;
				log_likelihood += (log(_weights(class_id, feature_id) + 1)
								   - log(_count_feature_class(class_id, 0) + N_FEATURES));
			}
		}

		double log_posterior = log_likelihood + log(prior);

		log_proba.push_back(log_posterior);
	}

	if (norm) log_proba = normalize(log_proba);
	return log_proba;
}

std::vector<double> normalize(std::vector<double> v) {
	double sum = 0;
	for (double i : v) {
		sum += i;
	}

	for (double &i : v) {
		i /= sum;
	}

	return v;
}

MultinominalNaiveBayes::MultinominalNaiveBayes() = default;


