//
// Created by nhatminh2947 on 10/10/19.
//

#include "GaussianNaiveBayes.h"

void GaussianNaiveBayes::fit(std::vector<Matrix> X, std::vector<int> y) {
	_train_size = X.size();
	Matrix mu[10];
	Matrix sigma[10];

	for (int id = 0; id < _train_size; ++id) {
		_prior(y[id], 0)++;
	}
	_prior = _prior / _train_size;

	for (int i = 0; i < N_CLASSES; ++i) {
		mu[i] = sigma[i] = Matrix(28, 28);
	}

	for (int i = 0; i < _train_size; ++i) {
		mu[y[i]] = mu[y[i]] + X[i];
	}

	for (int i = 0; i < N_CLASSES; ++i) {
		mu[i] = mu[i] / _train_size;
		_mean[i] = mu[i].flat();
	}

	for (int i = 0; i < _train_size; ++i) {
		Matrix A = X[i] - mu[y[i]];
		sigma[y[i]] = sigma[y[i]] + A.power(2);
	}

	for (int k = 0; k < N_CLASSES; ++k) {
		double min_sigma = INT64_MAX;
		for (int i = 0; i < N_ROWS; ++i) {
			for (int j = 0; j < N_COLS; ++j) {
				if (sigma[k](i, j) != 0) {
					min_sigma = std::min(min_sigma, sigma[k](i, j));
				}
			}
		}

		for (int i = 0; i < N_ROWS; ++i) {
			for (int j = 0; j < N_COLS; ++j) {
				if (sigma[k](i, j) == 0) {
					sigma[k](i, j) = min_sigma;
				}
			}
		}
	}

	for (int i = 0; i < N_CLASSES; ++i) {
		sigma[i] = sigma[i] / _train_size;
		_variance[i] = sigma[i].flat();
	}
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
		for (int j = 0; j < 784; ++j) {
//			double v = (_variance[i](0, j) == 0) ? 100 : _variance[i](0, j);
			probability += log(1.0 / (sqrt(_variance[i](0, j) * 2 * M_PI))) - (0.5 * pow(x(0, j) - _mean[i](0, j), 2) / _variance[i](0, j));
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