//
// Created by nhatminh2947 on 11/4/19.
//

#ifndef HOMEWORK_METRICS_H
#define HOMEWORK_METRICS_H


#include <vector>
#include "matrix.h"

class ConfusionMatrix {
private:
    Matrix matrix;
public:
    ConfusionMatrix(Row y_true, Row y_pred, int n_classes);

    friend std::ostream &operator<<(std::ostream &out, const ConfusionMatrix &matrix);

    friend std::ostream &operator<<(std::ostream &out, ConfusionMatrix &matrix);
};


#endif //HOMEWORK_METRICS_H
