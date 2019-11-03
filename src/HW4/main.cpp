//
// Created by nhatminh2947 on 11/3/19.
//

#include <GaussianDataGenerator.h>
#include <iostream>
#include <Dataset.h>

Dataset GenerateData(double n, double mx, double vx, double my, double vy) {
    GaussianDataGenerator dx(mx, vx);
    GaussianDataGenerator dy(my, vy);

    Dataset dataset;

    for (int i = 0; i < n; ++i) {
        dataset.Add(Point(dx.generate(), dy.generate()));
    }

    return dataset;
}

int main(int argc, const char *argv[]) {
    int n;
    Dataset d[2];

    std::cout << "Number of data points: ";
    std::cin >> n;
    for (int i = 0; i < 2; ++i) {
        double mx, vx, my, vy;
        std::cout << "mean x" << i << ": ";
        std::cin >> mx;
        std::cout << "variance x" << i << ": ";
        std::cin >> vx;
        std::cout << "mean y" << i << ": ";
        std::cin >> my;
        std::cout << "variance y" << i << ": ";
        std::cin >> vy;

        d[i] = GenerateData(n, mx, vx, my, vy);
    }



    return 0;
}
