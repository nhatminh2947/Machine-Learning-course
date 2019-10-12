//
// Created by nhatminh2947 on 10/8/19.
//

#include "NaiveBayesClassifier.h"

void NaiveBayesClassifier::test(std::vector<Matrix> test_images, std::vector<int> test_labels) {

}

void NaiveBayesClassifier::train(std::vector<Matrix> train_images, std::vector<int> train_labels) {
	std::vector<Matrix> train_image_labels[10];

	for (int i = 0; i < train_images.size(); ++i) {
		train_image_labels[train_labels[i]].emplace_back(train_images[i]);
	}

	Matrix X = train_images[0];
	for (int k = 0; k < 10; ++k) {
		double p_X_theta = 0.0;
		double p_X = 1.0;
		double p_theta = log(1.0 * train_image_labels[k].size() / 10000.0);

		int total = 0;
		for (int i = 0; i < 28; ++i) {
			for (int j = 0; j < 28; ++j) {
				for (auto &image : train_image_labels[k]) {
					total += (image(i, j) == X(i, j));
				}
			}
		}

		for (int i = 0; i < 28; ++i) {
			for (int j = 0; j < 28; ++j) {
				int count = 0;
				for (auto &image : train_image_labels[k]) {
					count += (image(i, j) == X(i, j));
				}

				p_X_theta += log((1.0 * count + 1) / (total + 28 * 28));
			}
		}

		std::cout << (p_X_theta) << std::endl;
		std::cout << (p_theta) << std::endl;
//        std::cout << (p_X) << std::endl;
		std::cout << k << ": " << ((p_X_theta + p_theta)) << std::endl;
	}
}

NaiveBayesClassifier::NaiveBayesClassifier(int mode) {
	_mode = mode;
}

std::vector<Matrix> NaiveBayesClassifier::preprocess_data(std::vector<Matrix> images) {
	for (int k = 0; k < images.size(); ++k) {
		for (int i = 0; i < N_ROWS; ++i) {
			for (int j = 0; j < N_COLS; ++j) {
				if(_mode == 0) {

				}
			}
		}
	}
}
