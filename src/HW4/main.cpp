//
// Created by nhatminh2947 on 11/3/19.
//

#include <GaussianDataGenerator.h>
#include <iostream>
#include <Dataset.h>

Dataset<2> GenerateData(double n, double mx[], double vx[], double my[], double vy[]) {
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

    Dataset<2> dataset;

    for (int i = 0; i < 2 * n; ++i) {
        int id = distribution(generator);

        if (count[id] == n) {
            id = 1 - id;
        }

        dataset.Add(Point<double, 2>(dx[id].generate(), dy[id].generate()), double(id));

        count[id]++;
    }

    return dataset;
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

    Dataset<2> dataset = GenerateData(n, mx, vx, my, vy);

    std::cout << dataset.size() << std::endl;

    return 0;
}
