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
    explicit LogisticRegression(Matrix X, std::array<int> y);
    void NewtonMethod();
    void GradientDescent();
};

template <typename GradType>
LogisticRegression<GradType>::LogisticRegression(Dataset<n_dimensions> d) {

}

#endif //HOMEWORK_LOGISTICREGRESSION_H
