//
// Created by nhatminh2947 on 11/4/19.
//

#include "ConfusionMatrix.h"

std::ostream &operator<<(std::ostream &out, const ConfusionMatrix &matrix) {
    out << "Confusion Matrix:";
    out << "            ";
    for (int k = 0; k < matrix.labels_; ++k) {
        out << " Predict cluster " << k;
    }
    out << std::endl;
    for (int i = 0; i < matrix.getRows(); ++i) {
        out << "Is cluster " << i;
        for (int j = 0; j < matrix.getCols(); ++j) {
            out << "\t\t" << matrix(i, j) << "\t\t";
        }
        out << std::endl;
    }

    return out;
}

std::ostream &operator<<(std::ostream &out, ConfusionMatrix &matrix) {
    for (int i = 0; i < matrix.matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            out << matrix(i, j) << " ";
        }
        out << std::endl;
    }

    return out;
}

ConfusionMatrix::ConfusionMatrix(Row y_true, Row y_pred, int n_classes) {
    for (int i = 0; i < y_true.size(); ++i) {
        data[int(y_true[i])][int(y_pred[i])]++;
    }
}
