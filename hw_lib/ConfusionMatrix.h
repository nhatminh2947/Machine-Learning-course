//
// Created by nhatminh2947 on 11/4/19.
//

#ifndef HOMEWORK_METRICS_H
#define HOMEWORK_METRICS_H


#include <vector>
#include "matrix.h"

class ConfusionMatrix {
private:
    Matrix<int> matrix_;
    int n_classes_;
public:
    ConfusionMatrix(Col<int> y_true, Col<int> y_pred, int n_classes);

    friend std::ostream &operator<<(std::ostream &out, const ConfusionMatrix &matrix);

    friend std::ostream &operator<<(std::ostream &out, ConfusionMatrix &matrix);
};


#endif //HOMEWORK_METRICS_H
