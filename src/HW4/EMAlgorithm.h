//
// Created by nhatminh2947 on 11/18/19.
//

#ifndef HOMEWORK_EMALGORITHM_H
#define HOMEWORK_EMALGORITHM_H


#include <Col.h>

class EMAlgorithm {
private:
    Col<double> weight_;
    Col<double> probability_;
    int n_class_;
    double *res_zeros{};
    double *res_ones{};

public:
    EMAlgorithm(int n_class = 10);

    int CountOnes(Matrix<double> image);

    int CountZeros(Matrix<double> image);

    void Maximization(Matrix<double> image);

    void Expectation(std::vector<Matrix<double>> image);
};


#endif //HOMEWORK_EMALGORITHM_H
