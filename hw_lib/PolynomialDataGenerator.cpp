//
// Created by nhatminh2947 on 10/23/19.
//

#include "PolynomialDataGenerator.h"
#include "GaussianDataGenerator.h"

double PolynomialDataGenerator::generate() {
    std::default_random_engine generator;
    Matrix<double> X(_n, 1);
    double x = distribution(generator);
    for (int i = 0; i < _n; ++i) {
        X(i, 0) = pow(x, i);
    }

    return (_W.T() * X)(0, 0) + _gdg.generate();
}

double PolynomialDataGenerator::generate(double x) {
    std::default_random_engine generator;
    Matrix<double> X(_n, 1);
    for (int i = 0; i < _n; ++i) {
        X(i, 0) = pow(x, i);
    }

    return (_W.T() * X)(0, 0) + _gdg.generate();
}

PolynomialDataGenerator::PolynomialDataGenerator(int n, double a, Matrix<double> W) : _W(W) {
    distribution = std::uniform_real_distribution<double>(-1, 1);
    _n = n;
    _gdg = GaussianDataGenerator(0, a);
}
