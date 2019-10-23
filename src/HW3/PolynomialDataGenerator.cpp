//
// Created by nhatminh2947 on 10/23/19.
//

#include "PolynomialDataGenerator.h"
#include "GaussianDataGenerator.h"

double PolynomialDataGenerator::generate() {
    std::default_random_engine generator;
    Matrix X(_n, 1);
    double x = distribution(generator);
    for (int i = 0; i < _n; ++i) {
        X(i, 0) = pow(x, i);
    }

    return sum(_W.T() * X) + _gdg.generate();
}

PolynomialDataGenerator::PolynomialDataGenerator(int n, double a, Matrix W) {
    distribution = std::uniform_real_distribution<double>(-1, 1);
    _W = W;
    _n = n;
    _gdg = GaussianDataGenerator(0, a);
}
