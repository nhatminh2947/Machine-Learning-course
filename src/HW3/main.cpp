//
// Created by nhatminh2947 on 10/23/19.
//

#include <iostream>
#include <matrix.h>
#include "GaussianDataGenerator.h"
#include "PolynomialDataGenerator.h"

void hw11(double mean, double var) {
    GaussianDataGenerator gdg(mean, var);

    int count[1000];
    for (int i = 0; i < 1000; ++i) {
        count[i] = 0;
    }

    double number;
    for (int i = 0; i < 100000; ++i) {
        number = gdg.generate();

        count[int((number + 3) * 10)]++;
//        std::cout << number << std::endl;
    }

    for (int i = 0; i < 60; ++i) {
        printf("%5.2f ", (1.0 * i - 30) / 10);
    }
    std::cout << std::endl;

    for (int i = 0; i < 60; ++i) {
        printf("%5d ", count[i]);
    }
    std::cout << std::endl;
}

void hw12(int n, double a) {
    Matrix W(n, 1);

    for (int i = 0; i < n; ++i) {
        std::cin >> W(i, 0);
    }

    std::cout << W << std::endl;

    PolynomialDataGenerator pdg(n, a, W);

    int count[1000];
    for (int &i : count) {
        i = 0;
    }

    double number;
    for (int i = 0; i < 100000; ++i) {
        number = pdg.generate();

        count[int((number + 3) * 10)]++;
    }

    for (int i = 0; i < 60; ++i) {
        printf("%5.2f ", (1.0 * i - 30) / 10);
    }
    std::cout << std::endl;

    for (int i = 0; i < 60; ++i) {
        printf("%5d ", count[i]);
    }
    std::cout << std::endl;
}

int main(int argc, const char *argv[]) {
    int hw = 12;
    double mean = 0;
    double var = 0.2;
    int n = 2;
    double a = 1;
    std::vector<double> w;

    for (int i = 1; i < argc; i++) {
        std::string para(argv[i]);
        if (para.find("--hw=") == 0) {
            hw = std::stoi(para.substr(para.find('=') + 1));
        } else if (para.find("--mean=") == 0) {
            mean = std::stoi(para.substr(para.find('=') + 1));
        } else if (para.find("--var=") == 0) {
            var = std::stoi(para.substr(para.find('=') + 1));
        }
    }

    if (hw == 11) {
        hw11(mean, var);
    } else if (hw == 12) {
        hw12(n, a);
    }


}