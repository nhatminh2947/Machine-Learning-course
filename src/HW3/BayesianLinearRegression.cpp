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
    double a = 3;
    double b = 1;
    int n = 3;

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
    Matrix covar = (1 / b) * IdentityMatrix(n);

    for (int i = 0; i < n; ++i) {
        std::cout << "w[" << i << "] = ";
        std::cin >> W(i, 0);
    }

    PolynomialDataGenerator pdg(n, a, W);


    std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(-1, 1);
    std::default_random_engine generator;

    std::ofstream file_weight("../src/HW3/weights.txt", std::ios::trunc | std::ios::out);
    remove("../src/HW3/data.txt");
    std::ofstream file_data("../src/HW3/data.txt", std::ios::app | std::ios::out);
    std::cout << std::setprecision(10);
    Matrix covar_prev = covar;
    Matrix mean_prev = mean;
    for (int i = 0; i < 10000; ++i) {
        file_weight << n << std::endl;
        double x = distribution(generator);
        double y = pdg.generate(x);
        file_data << std::setprecision(10) << x << " " << y << std::endl;

        std::cout << "Add data point (" << x << ", " << y << "):" << std::endl << std::endl;

        Matrix X = Matrix::ToDesignMatrix(x, n);
        covar = (covar_prev.inverse() + a * (X * X.T())).inverse();
        mean = covar * (covar_prev.inverse() * mean_prev + a * y * X);

        std::cout << "Posterior mean:" << std::endl;
        std::cout << mean << std::endl;
        file_weight << std::setprecision(10) << mean;
        std::cout << "Posterior variance:" << std::endl;
        std::cout << covar << std::endl;
        file_weight << std::setprecision(10) << covar << std::endl;

        std::cout << "Predictive distribution ~ N(" << sum(mean.T() * X) << ", " << a + sum(X.T() * covar * X)
                  << ")" << std::endl;
        std::cout << "--------------------------------------------------\n";

        usleep(100000);
        file_weight.seekp(std::ios::beg);

        covar_prev = covar;
        mean_prev = mean;
    }

    file_data.close();
    file_weight.close();
}