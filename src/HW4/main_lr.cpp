//
// Created by nhatminh2947 on 11/3/19.
//

#include <GaussianDataGenerator.h>
#include <iostream>
#include <Point.h>
#include <ConfusionMatrix.h>
#include <iomanip>
#include <fstream>
#include "LogisticRegression.h"

void GenerateData(double n, double mx[], double vx[], double my[], double vy[], Matrix<double> &X, Col<int> &y) {
    std::random_device rd;
    std::default_random_engine generator{rd()};
    std::uniform_int_distribution<int> distribution(0, 1);

    GaussianDataGenerator dx[2];
    GaussianDataGenerator dy[2];

    int count[2];
    for (int i = 0; i < 2; ++i) {
        count[i] = 0;
        dx[i] = GaussianDataGenerator(mx[i], vx[i]);
        dy[i] = GaussianDataGenerator(my[i], vy[i]);
    }

    std::ofstream lr_data("../src/HW4/lr_data.txt");

    for (int i = 0; i < 2 * n; ++i) {
        int id = distribution(generator);

        if (count[id] == n) {
            id = 1 - id;
        }

        X(i, 0) = 1;
        X(i, 1) = dx[id].generate();
        X(i, 2) = dy[id].generate();

        y[i] = (double) id;

        count[id]++;

        lr_data << X(i, 1) << " " << X(i, 2) << " " << y[i] << std::endl;
    }
    lr_data.close();

}

int main(int argc, const char *argv[]) {
    int n;

    std::cout << "Number of data points: ";
    std::cin >> n;
    double mx[2], vx[2], my[2], vy[2];

    for (int i = 0; i < 2; ++i) {
        std::cout << "mean x" << i + 1 << ": ";
        std::cin >> mx[i];
        std::cout << "variance x" << i + 1 << ": ";
        std::cin >> vx[i];
        std::cout << "mean y" << i + 1 << ": ";
        std::cin >> my[i];
        std::cout << "variance y" << i + 1 << ": ";
        std::cin >> vy[i];
    }

    Matrix<double> X(2 * n, 3);
    Col<int> y_true(2 * n);

    GenerateData(n, mx, vx, my, vy, X, y_true);
    std::cout << "Gradient descent:" << std::endl;
    LogisticRegression<double> lr_gradient(X, Col<double>(y_true), 0.01, 1000, true);
    Col<int> y_pred = lr_gradient.Classify(X);

    ConfusionMatrix cm_gradient(y_true, y_pred, 2);

    std::ofstream file_lr_gradient("../src/HW4/lr_gradient.txt");
    for (int i = 0; i < 2 * n; ++i) {
        file_lr_gradient << X(i, 1) << " " << X(i, 2) << " " << y_pred[i] << std::endl;
    }
    file_lr_gradient.close();

    std::cout << cm_gradient << std::endl;

    std::cout << "Sensitivity (Successfully predict cluster 1): " << std::setprecision(5) << std::fixed
              << cm_gradient.sensitivity() << std::endl;
    std::cout << "Specificity (Successfully predict cluster 2): " << std::setprecision(5) << std::fixed
              << cm_gradient.specificity() << std::endl;

    std::cout << "\n\n----------------------------------------" << std::endl;


    std::cout << "Newton's method:" << std::endl;

    LogisticRegression<double> lr_newton(X, Col<double>(y_true), 0.01, 1000, false);
    y_pred = lr_newton.Classify(X);

    ConfusionMatrix cm_newton(y_true, y_pred, 2);

    std::ofstream file_lr_newton("../src/HW4/lr_newton.txt");
    for (int i = 0; i < 2 * n; ++i) {
        file_lr_newton << X(i, 1) << " " << X(i, 2) << " " << y_pred[i] << std::endl;
    }
    file_lr_newton.close();

    std::cout << cm_newton << std::endl;

    std::cout << "Sensitivity (Successfully predict cluster 1): " << std::setprecision(5) << std::fixed
              << cm_newton.sensitivity() << std::endl;
    std::cout << "Specificity (Successfully predict cluster 2): " << std::setprecision(5) << std::fixed
              << cm_newton.specificity() << std::endl;

    return 0;
}
