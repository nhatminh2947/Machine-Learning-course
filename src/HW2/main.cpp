//
// Created by nhatminh2947 on 10/8/19.
//

#include "IdxFileReader.h"
#include "MultinomialNaiveBayes.h"
#include "GaussianNaiveBayes.h"
#include "OnlineLearning.h"
#include <cmath>
#include <iomanip>
#include <memory>

Matrix bin_image(Matrix image) {
	Matrix result(28, 28);
	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			result(i, j) = int(image(i, j)) / 8;
		}
	}

	return result;
}

std::vector<Matrix> ExtractFeatures(const std::vector<Matrix> &images) {
	std::vector<Matrix> result;
	result.reserve(images.size());

	for (auto image : images) {
		result.push_back(bin_image(image));
	}

	return result;
}

void hw1(int mode) {
	std::vector<Matrix> test_images = IdxFileReader::ReadImages("../src/HW2/t10k-images-idx3-ubyte");
	std::vector<int> test_labels = IdxFileReader::ReadLabels("../src/HW2/t10k-labels-idx1-ubyte");
	std::vector<Matrix> train_images = IdxFileReader::ReadImages("../src/HW2/train-images-idx3-ubyte");
	std::vector<int> train_labels = IdxFileReader::ReadLabels("../src/HW2/train-labels-idx1-ubyte");

	std::unique_ptr<NaiveBayes> classifier;

	if (mode == 0) {
		train_images = ExtractFeatures(train_images);
		test_images = ExtractFeatures(test_images);
		classifier = std::make_unique<MultinomialNaiveBayes>();
	} else {
		classifier = std::make_unique<GaussianNaiveBayes>();
	}

	classifier->fit(train_images, train_labels);

	int error_rate = 0;

	for (int i = 0; i < 10000; ++i) {
		std::cout << "Postirior (in log scale):" << std::endl;
		int test_image_id = i;
		int prediction = classifier->predict(test_images[test_image_id]);
		std::cout << "Prediction: " << prediction << ", Ans: " << test_labels[test_image_id] << std::endl << std::endl;

		error_rate += (prediction != test_labels[test_image_id]);
	}

	std::cout << "Imagination of numbers in Bayesian classifier:\n" << std::endl;
//	for (int class_id = 0; class_id < 10; ++class_id) {
//		std::cout << class_id << ":" << std::endl;
//		for (int i = 0; i < 10000; ++i) {
//			if (test_labels[i] == class_id) {
//				for (int j = 0; j < 28; ++j) {
//					for (int k = 0; k < 28; ++k) {
//						std::cout << ((test_images[i](j, k) < (mode ? 128 : 16)) ? 0 : 1) << " ";
//					}
//					std::cout << std::endl;
//				}
//				std::cout << std::endl;
//				break;
//			}
//		}
//	}

	classifier->imagination();

	std::cout << "Error rate: " << error_rate * 1.0 / 10000 << " (" << error_rate << "/" << "10000)" << std::endl;
}

void hw2(int a, int b) {
	OnlineLearning onlineLearning(a, b);
	std::ifstream input("../src/HW2/testfile.txt");
	std::string event;
	int cs = 1;
	while (input >> event) {
		std::cout << std::setprecision(17) << "Case " << cs << ": " << event << std::endl;
		int prior_a = onlineLearning.getA();
		int prior_b = onlineLearning.getB();
		std::cout << "Likelihood: " << onlineLearning.Event(event) << std::endl;
		std::cout << "Beta prior: \ta = " << prior_a << "\tb = " << prior_b << std::endl;
		std::cout << "Beta posterior: a = " << onlineLearning.getA() << "\tb = " << onlineLearning.getB() << std::endl;
		std::cout << std::endl;
	}
}

int main(int argc, const char *argv[]) {
	int hw = 1;
	int mode = 0;
	int a = 0;
	int b = 0;
	for (int i = 1; i < argc; i++) {
		std::string para(argv[i]);
		if (para.find("--hw=") == 0) {
			hw = std::stoi(para.substr(para.find('=') + 1));
		} else if (para.find("--mode=") == 0) {
			mode = std::stoi(para.substr(para.find('=') + 1));
		} else if (para.find("--a=") == 0) {
			a = std::stoi(para.substr(para.find('=') + 1));
		} else if (para.find("--b=") == 0) {
			b = std::stoi(para.substr(para.find('=') + 1));
		}
	}

	if (hw == 1) {
		hw1(mode);
	} else {
		hw2(a, b);
	}

	return 0;
}
