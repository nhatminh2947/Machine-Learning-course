//
// Created by nhatminh2947 on 10/10/19.
//

#ifndef MACHINE_LEARNING_COURSE_GAUSSIANNAIVEBAYES_H
#define MACHINE_LEARNING_COURSE_GAUSSIANNAIVEBAYES_H


#include <vector>
#include <matrix.h>
#include "NaiveBayes.h"
#include <cmath>

class GaussianNaiveBayes : public NaiveBayes{
private:
    Matrix _mean[10]{};
    Matrix _variance[10]{};
public:
    GaussianNaiveBayes();

    void fit(std::vector<Matrix> X, std::vector<int> y) override;

    int predict(Matrix x) override;

    std::vector<double> predict_log_proba(Matrix x, bool normalize) override;
};


#endif //MACHINE_LEARNING_COURSE_GAUSSIANNAIVEBAYES_H
