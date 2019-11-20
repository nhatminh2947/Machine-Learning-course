//
// Created by nhatminh2947 on 10/8/19.
//

#include "IdxFileReader.h"


int IdxFileReader::reverse(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int) c1 << 24U) + ((int) c2 << 16U) + ((int) c3 << 8U) + c4;
}

Matrix<double> IdxFileReader::ReadImages(const std::string &path, int n_data) {
    std::ifstream file(path, std::ios::in | std::ios::binary);

    if (!file.is_open()) {
        throw std::invalid_argument("File not exist!");
    }

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows;
    int n_cols;
    unsigned char pixel;

    file.read((char *) &magic_number, sizeof(magic_number));
    file.read((char *) &number_of_images, sizeof(number_of_images));
    file.read((char *) &n_rows, sizeof(n_rows));
    file.read((char *) &n_cols, sizeof(n_cols));

    number_of_images = reverse(number_of_images);
    n_rows = reverse(n_rows);
    n_cols = reverse(n_cols);

    Matrix<double> images(std::min(number_of_images, n_data), 28 * 28);
    for (int k = 0; k < number_of_images && k < n_data; ++k) {
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                file.read((char *) &pixel, sizeof(pixel));

                images(k, i * 28 + j) = int(pixel);
            }
        }
    }

    return images.T();
}

std::vector<int> IdxFileReader::ReadLabels(const std::string &path, int n_data) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    std::vector<int> labels;

    if (!file.is_open()) {
        throw std::invalid_argument("File not exist!");
    }

    int magic_number = 0;
    int number_of_labels = 0;
    unsigned char pixel;

    file.read((char *) &magic_number, sizeof(magic_number));
    file.read((char *) &number_of_labels, sizeof(number_of_labels));

    number_of_labels = std::min(n_data, reverse(number_of_labels));

    for (int k = 0; k < number_of_labels; ++k) {
        file.read((char *) &pixel, sizeof(pixel));
        labels.push_back((int) pixel);
    }

    return labels;
}
