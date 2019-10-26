//
// Created by nhatminh2947 on 10/23/19.
//

#include <iostream>
#include <iomanip>
#include "GaussianDataGenerator.h"
#include "PolynomialDataGenerator.h"

int main(int argc, const char *argv[]) {
    double a = 1;
    double b = 1;
    int n = 4;

    for (int i = 1; i < argc; i++) {
        std::string para(argv[i]);
        if (para.find("--a=") == 0) {
            a = std::stod(para.substr(para.find('=') + 1));
        } else if (para.find("--b=") == 0) {
            b = std::stod(para.substr(para.find('=') + 1));
        } else if (para.find("--b=") == 0) {
            n = std::stoi(para.substr(para.find('=') + 1));
        }
    }

    Matrix W(n, 1);
    Matrix mean(n, 1);
    Matrix covar = (1 / a) * IdentityMatrix(n);

    GaussianDataGenerator gdg(0, 1 / b);

    for (int i = 0; i < n; ++i) {
        std::cout << "w[" << i << "] = ";
        std::cin >> W(i, 0);
    }

    PolynomialDataGenerator pdg(n, a, W);


    std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(-1, 1);
    std::default_random_engine generator;

    for (int i = 0; i < 1000000; ++i) {
        double x = distribution(generator);
        double y = pdg.generate(x);

        std::cout << "Add data point (" << x << ", " << y << "):" << std::endl << std::endl;

        Matrix X = Matrix::ToDesignMatrix(x, n);
        Matrix covar_prev = covar;
        Matrix mean_prev = mean;
        covar = (covar_prev.inverse() + b * X * X.T()).inverse();
        mean = covar * (covar_prev.inverse() * mean_prev + b * y * X);

        std::cout << "Posterior mean:" << std::endl;
        std::cout << mean << std::endl;

        std::cout << "Posterior variance:" << std::endl;
        std::cout << covar << std::endl;

        std::cout << "Predictive distribution ~ N(0.62305, 1.34848)" << std::endl;
        std::cout << "--------------------------------------------------\n";
    }
}