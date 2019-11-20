//
// Created by nhatminh2947 on 11/4/19.
//

#include "ConfusionMatrix.h"

std::ostream &operator<<(std::ostream &out, const ConfusionMatrix &confusion_matrix) {
    out << "Confusion Matrix:" << std::endl;
    out << "            ";
    for (int k = 0; k < confusion_matrix.n_classes_; ++k) {
        out << " Predict cluster " << k << "\t";
    }
    out << std::endl;
    for (int i = 0; i < confusion_matrix.n_classes_; ++i) {
        out << "Is cluster " << i;
        for (int j = 0; j < confusion_matrix.n_classes_; ++j) {
            out << "\t\t" << confusion_matrix.matrix_(i, j) << "\t\t\t";
        }
        out << std::endl;
    }

    return out;
}

std::ostream &operator<<(std::ostream &out, ConfusionMatrix &confusion_matrix) {
    out << "Confusion Matrix:" << std::endl;
    out << "            ";
    for (int k = 0; k < confusion_matrix.n_classes_; ++k) {
        out << " Predict cluster " << k << "\t";
    }
    out << std::endl;
    for (int i = 0; i < confusion_matrix.n_classes_; ++i) {
        out << "Is cluster " << i;
        for (int j = 0; j < confusion_matrix.n_classes_; ++j) {
            out << "\t\t" << confusion_matrix.matrix_(i, j) << "\t\t\t";
        }
        out << std::endl;
    }

    return out;
}

ConfusionMatrix::ConfusionMatrix(Col<int> y_true, Col<int> y_pred, int n_classes) : matrix_(n_classes) {
    n_classes_ = n_classes;
    for (int i = 0; i < y_true.size(); ++i) {
        matrix_(y_true[i], y_pred[i])++;
    }
}

double ConfusionMatrix::sensitivity() {
    return 1.0 * matrix_(0, 0) / (matrix_(0, 0) + matrix_(0, 1));
}

double ConfusionMatrix::specificity() {
    return 1.0 * matrix_(1, 1) / (matrix_(1, 1) + matrix_(1, 0));
}

ConfusionMatrix::ConfusionMatrix(std::vector<int> y_true, std::vector<int> y_pred, int n_classes) : matrix_(n_classes) {
    n_classes_ = n_classes;
    for (int i = 0; i < y_true.size(); ++i) {
        matrix_(y_true[i], y_pred[i])++;
    }
}
