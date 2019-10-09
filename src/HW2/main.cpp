//
// Created by nhatminh2947 on 10/8/19.
//

#include "IdxFileReader.h"
#include <cmath>

Matrix counting(Matrix image) {
    Matrix count(32, 1);

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            count(int(image(i, j)) / 8, 0) += 1.0;
        }
    }

    return count;
}

std::vector<Matrix> convert_discrete_mode(const std::vector<Matrix> &images) {
    std::vector<Matrix> result;
    result.reserve(images.size());

    for (auto image : images) {
        result.emplace_back(counting(image));
    }

    return result;
}

int main() {
    int test_size = 10000;
    std::vector<Matrix> test_images = IdxFileReader::ReadImages("../src/HW2/t10k-images-idx3-ubyte");
    std::vector<int> test_labels = IdxFileReader::ReadLabels("../src/HW2/t10k-labels-idx1-ubyte");
    std::vector<Matrix> test_images_discrete;

    std::vector<Matrix> test_image_labels[10];

    double p_C[10];
    double counting_X_C[10][32];
    double p_X_C[10][32];

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 32; ++j) {
            p_X_C[i][j] = counting_X_C[i][j] = 0;
        }
    }

    for (int i = 0; i < test_size; ++i) {
        test_image_labels[test_labels[i]].emplace_back(test_images[i]);
        p_C[test_labels[i]]++;
    }

    for (int i = 0; i < 10; ++i) {
        p_C[i] = p_C[i] / test_size;

        std::cout << "Class " << i << " probability: " << p_C[i] << std::endl;
    }

    test_images_discrete = convert_discrete_mode(test_images);

    int image_id = 2;

    Matrix X = test_images_discrete[image_id];

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            std::cout << (test_images[image_id](i, j) <= 128 ? 0 : 1);
        }
        std::cout << std::endl;
    }

    double p[10];
    double marginal = 0;

    for (int k = 0; k < 10; ++k) {
        for (int j = 0; j < 32; ++j) {
            for (int i = 0; i < test_size; ++i) {
                if (test_labels[i] == k) {
                    counting_X_C[k][j] += test_images_discrete[i](j, 0);
                }
            }
        }
    }

//    for (int k = 0; k < 10; ++k) {
//        for (int i = 0; i < 32; ++i) {
//            std::cout << counting_X_C[k][i] <<  " ";
//        }
//        std::cout << std::endl;
//    }

    for (int k = 0; k < 10; ++k) {
        double total = 0;
        for (int i = 0; i < 32; ++i) {
            total += counting_X_C[k][i];
        }

//        std::cout << "total: " << total << std::endl;

        for (int i = 0; i < 32; ++i) {
            p_X_C[k][i] = (counting_X_C[k][i] + 1) / (32 + total);

//            std::cout << p_X_C[k][i] << " ";
        }
//        std::cout << std::endl;
    }

    for (int k = 0; k < 10; ++k) {
        double p_C_X = 0;
        for (int i = 0; i < 32; ++i) {
            p_C_X += X(i, 0) * log(p_X_C[k][i]);
        }

        p_C_X += log(p_C[k]);
        marginal += p_C_X;
    }

    for (int k = 0; k < 10; ++k) {
        double p_C_X = 0;
        for (int i = 0; i < 32; ++i) {
            p_C_X += X(i, 0) * log(p_X_C[k][i]);
        }

        p_C_X += log(p_C[k]);
        p_C_X /= marginal;
        std::cout << k << ": " << p_C_X << std::endl;
    }

    int ans = 0;
    std::cout << "Prediction: " << ans << ", Ans: " << test_labels[image_id] << std::endl;

    return 0;
}