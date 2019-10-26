//
// Created by nhatminh2947 on 10/23/19.
//

#ifndef HOMEWORK_POLYNOMIALDATAGENERATOR_H
#define HOMEWORK_POLYNOMIALDATAGENERATOR_H


#include <matrix.h>
#include <random>
#include "GaussianDataGenerator.h"

class PolynomialDataGenerator {
private:
    int _n;
    Matrix _W{};
    std::uniform_real_distribution<double> distribution;
    GaussianDataGenerator _gdg = GaussianDataGenerator(0, 0);

public:
    PolynomialDataGenerator(int n, double a, Matrix W);

    double generate();

    double generate(double x);
};


#endif //HOMEWORK_POLYNOMIALDATAGENERATOR_H
