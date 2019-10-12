//
// Created by nhatminh2947 on 10/8/19.
//

#include "IdxFileReader.h"
#include "MultinomialNaiveBayes.h"
#include "GaussianNaiveBayes.h"
#include <cmath>

Matrix bin_image(Matrix image) {
    Matrix result(28, 28);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            result(i, j)  = int(image(i, j)) / 8;
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

int main() {
    std::vector<Matrix> test_images = IdxFileReader::ReadImages("../src/HW2/t10k-images-idx3-ubyte");
    std::vector<int> test_labels = IdxFileReader::ReadLabels("../src/HW2/t10k-labels-idx1-ubyte");
    std::vector<Matrix> train_images = IdxFileReader::ReadImages("../src/HW2/train-images-idx3-ubyte");
    std::vector<int> train_labels = IdxFileReader::ReadLabels("../src/HW2/train-labels-idx1-ubyte");

    std::vector<Matrix> train_image_features = ExtractFeatures(train_images);
    std::vector<Matrix> test_image_features = ExtractFeatures(test_images);

//    MultinomialNaiveBayes classifier;
    GaussianNaiveBayes classifier;
    classifier.fit(train_images, train_labels);
//	classifier.fit(train_image_features, train_labels);
	std::cout << "Completed Training" << std::endl;
    int error_rate = 0;

    for (int i = 0; i < 10000; ++i) {
        std::cout << "Test " << i << std::endl;
        int test_image_id = i;
//        int prediction = classifier.predict(test_image_features[test_image_id]);
		int prediction = classifier.predict(test_images[test_image_id]);
		if(prediction == -1) {
			break;
		}

		std::cout << "Prediction: " << prediction << ", Ans: " << test_labels[test_image_id] << std::endl;

        error_rate += (prediction != test_labels[test_image_id]);
    }

    std::cout << "Error rate: " << error_rate * 1.0 / 10000 << " (" << error_rate << "/" << "10000)" << std::endl;

    return 0;
}
