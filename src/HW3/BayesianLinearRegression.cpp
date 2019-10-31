//
// Created by nhatminh2947 on 10/23/19.
//

#include <iostream>
#include <iomanip>
#include <fstream>
#include <zconf.h>
#include "GaussianDataGenerator.h"
#include "PolynomialDataGenerator.h"

int main(int argc, const char *argv[]) {
    double a = 1;
    double b = 100;
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

    double beta = 1.0 / a;

    Matrix W(n, 1);
    Matrix mean(n, 1);
    Matrix covar = (1 / b) * IdentityMatrix(n);

    std::ofstream file_weight("../src/HW3/weights_0.txt", std::ios::trunc | std::ios::out);
    for (int i = 0; i < n; ++i) {
        std::cout << "w[" << i << "] = ";
        std::cin >> W(i, 0);
        file_weight << std::setprecision(10) << W(i, 0) << std::endl;
    }
    file_weight << std::setprecision(10) << a << std::endl;
    file_weight.close();

    PolynomialDataGenerator pdg(n, a, W);

    std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(-1, 1);
    std::default_random_engine generator;

    remove("../src/HW3/data.txt");
    std::ofstream file_data("../src/HW3/data.txt", std::ios::app | std::ios::out);

    std::vector<double> xs;
    std::vector<double> ys;

    int n_obser = 100000;
    for (int i = 0; i < n_obser; ++i) {
        double x = distribution(generator);
        double y = pdg.generate(x);
        file_data << std::setprecision(10) << x << " " << y << std::endl;


        xs.emplace_back(x);
        ys.emplace_back(y);
    }
    file_data.close();

    std::cout << std::setprecision(10);
    Matrix covar_prev = covar;
    Matrix mean_prev = mean;
    for (int i = 1; i <= n_obser; ++i) {
        double x = xs[i - 1];
        double y = ys[i - 1];
        std::cout << "Add data point (" << x << ", " << y << "):" << std::endl << std::endl;

        Matrix X = Matrix::ToDesignMatrix(x, n);
        covar = (covar_prev.inverse() + beta * (X * X.T())).inverse();
        mean = covar * (covar_prev.inverse() * mean_prev + beta * y * X);

        if (i == 10 || i == 50 || i == 10000) {
            std::ofstream file_weight("../src/HW3/weights_" + std::to_string(i) + ".txt", std::ios::trunc | std::ios::out);
            file_weight << n << std::endl;
            file_weight << std::setprecision(10) << mean;
            file_weight << std::setprecision(10) << covar << std::endl;
            file_weight.close();
        }

        std::cout << "Posterior mean:" << std::endl;
        std::cout << mean << std::endl;
        std::cout << "Posterior variance:" << std::endl;
        std::cout << covar << std::endl;

        std::cout << "Predictive distribution ~ N(" << sum(mean.T() * X) << ", " << 1 / beta + sum(X.T() * covar * X)
                  << ")" << std::endl;
        std::cout << "--------------------------------------------------\n";

        covar_prev = covar;
        mean_prev = mean;
    }
}