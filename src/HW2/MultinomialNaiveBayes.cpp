//
// Created by nhatminh2947 on 10/8/19.
//

#include <cmath>
#include "MultinomialNaiveBayes.h"

void MultinomialNaiveBayes::fit(std::vector<Matrix> X, std::vector<int> y) {
	_train_size = X.size();
	for (int id = 0; id < _train_size; ++id) {
		_prior(y[id], 0)++;
		for (int row_id = 0; row_id < N_ROWS; ++row_id) {
			for (int col_id = 0; col_id < N_COLS; ++col_id) {
				int feature_id = int(X[id](row_id, col_id)) * N_ROWS * N_COLS + row_id * N_ROWS + col_id;
				_weights(y[id], feature_id) += 1;
			}
		}
	}

	for (int i = 0; i < N_CLASSES; ++i) {
		_prior(i, 0) = _prior(i, 0) / _train_size;
		for (int row_id = 0; row_id < N_ROWS; ++row_id) {
			for (int col_id = 0; col_id < N_COLS; ++col_id) {
				for (int bin = 0; bin < 32; ++bin) {
					_count_feature_class(i, 0) += _weights(i, bin * N_ROWS * N_COLS + row_id * N_ROWS + col_id);
				}
			}
		}

	}
}

int MultinomialNaiveBayes::predict(Matrix x) {
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

std::vector<double> MultinomialNaiveBayes::predict_log_proba(Matrix x, bool norm) {
	std::vector<double> log_proba;

	for (int class_id = 0; class_id < N_CLASSES; ++class_id) {
		double log_likelihood = 0;
		for (int row_id = 0; row_id < 28; ++row_id) {
			for (int col_id = 0; col_id < 28; ++col_id) {
				int feature_id = int(x(row_id, col_id)) * 28 * 28 + row_id * 28 + col_id;
				log_likelihood += log((_weights(class_id, feature_id) + 1))
								  - log((_count_feature_class(class_id, 0) + N_FEATURES));
			}
		}


		double log_posterior = log_likelihood + log(_prior(class_id, 0));

		log_proba.push_back(log_posterior);
	}

	if (norm) log_proba = normalize(log_proba);
	return log_proba;
}

MultinomialNaiveBayes::MultinomialNaiveBayes() = default;


