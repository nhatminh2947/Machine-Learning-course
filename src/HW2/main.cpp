//
// Created by nhatminh2947 on 10/8/19.
//

#include "IdxFileReader.h"
#include <cmath>

int main() {
    int test_size = 10000;
    std::vector<Matrix> test_images = IdxFileReader::ReadImages("../src/HW2/t10k-images-idx3-ubyte");
    std::vector<int> test_labels = IdxFileReader::ReadLabels("../src/HW2/t10k-labels-idx1-ubyte");

    std::vector<Matrix> test_image_labels[10];

    for (int i = 0; i < test_size; ++i) {
        test_image_labels[test_labels[i]].emplace_back(test_images[i]);
    }

    int image_id = 0;

    std::cout << test_images[image_id] << std::endl;
    Matrix X = test_images[image_id];

    double p[10];
    double marginal = 0;

    for (int k = 0; k < 10; ++k) {

        double p_X_theta = 0.0;
        double p_X = 1.0;
        double p_theta = log(1.0 * test_image_labels[k].size() / 10000.0);

        int total = 0;
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                for (auto &image : test_image_labels[k]) {
                    total += (image(i, j) == X(i, j));
                }
            }
        }

        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                int count = 0;
                for (auto &image : test_image_labels[k]) {
                    count += (image(i, j) == X(i, j));
                }

                p_X_theta += log((1.0 * count + 1) / (total + 28 * 28));
            }
        }

        std::cout << (p_X_theta) << std::endl;
        std::cout << (p_theta) << std::endl;
//        std::cout << (p_X) << std::endl;
        std::cout << k << ": " << ((p_X_theta + p_theta)) << std::endl;

        p[k] = p_X_theta + p_theta;
        marginal += p[k];
    }

    std::cout << marginal << std::endl;
    for (int k = 0; k < 10; ++k) {
        std::cout << k << ": " << p[k] / marginal << std::endl;
    }

    int ans = 0;
    std::cout << "Prediction: " << ans << ", Ans: " << test_labels[image_id] << std::endl;

    return 0;
}