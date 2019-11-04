//
// Created by nhatminh2947 on 11/3/19.
//

#include <GaussianDataGenerator.h>
#include <iostream>
#include <Dataset.h>

void GenerateData(double n, double mx[], double vx[], double my[], double vy[], Matrix<double> &X, Row<int> &y) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 1);

    GaussianDataGenerator dx[2];
    GaussianDataGenerator dy[2];
    int count[2];
    for (int i = 0; i < 2; ++i) {
        count[i] = 0;
        dx[i] = GaussianDataGenerator(mx[i], vx[i]);
        dy[i] = GaussianDataGenerator(my[i], vy[i]);
    }

    for (int i = 0; i < 2 * n; ++i) {
        int id = distribution(generator);

        if (count[id] == n) {
            id = 1 - id;
        }

        X(i, 1) = dx[id].generate();
        X(i, 2) = dy[id].generate();

        y(i) = (double) id;

        count[id]++;
    }
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
    Row<int> y(2 * n);

    GenerateData(n, mx, vx, my, vy, X, y);

    std::cout << X << std::endl;

    return 0;
}
