//
// Created by nhatminh2947 on 10/23/19.
//

#include <chrono>
#include "GaussianDataGenerator.h"

GaussianDataGenerator::GaussianDataGenerator(double mean, double var) {
    _mean = mean;
    _var = var;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    distribution = std::uniform_real_distribution<double>(-1, 1);
}

double GaussianDataGenerator::generate() {
    double u;
    double v;
    double s;
    do {
        u = distribution(generator);
        v = distribution(generator);
        s = u * u + v * v;
    } while (s >= 1);

    double x = u * sqrt((-2 * log(s)) / s);

    return _mean + sqrt(_var) * x;
}
