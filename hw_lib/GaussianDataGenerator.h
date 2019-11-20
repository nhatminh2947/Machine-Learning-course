//
// Created by nhatminh2947 on 10/23/19.
//

#ifndef HOMEWORK_GAUSSIANDATAGENERATOR_H
#define HOMEWORK_GAUSSIANDATAGENERATOR_H

#include <cmath>
#include <random>

class GaussianDataGenerator {
private:
    double _mean{};
    double _var{};
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;

public:
    GaussianDataGenerator(double mean, double var);
    GaussianDataGenerator() = default;
    double generate();
};


#endif //HOMEWORK_GAUSSIANDATAGENERATOR_H
