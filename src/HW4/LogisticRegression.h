//
// Created by nhatminh2947 on 11/3/19.
//

#ifndef HOMEWORK_LOGISTICREGRESSION_H
#define HOMEWORK_LOGISTICREGRESSION_H


#include <Dataset.h>

template <typename GradType>
class LogisticRegression {
private:
    friend double sigmoid(double x);

public:
    explicit LogisticRegression(Matrix<double> X, Row<int> y);
    void NewtonMethod();
    void GradientDescent();
};

#endif //HOMEWORK_LOGISTICREGRESSION_H
