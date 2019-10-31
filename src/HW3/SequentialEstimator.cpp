//
// Created by nhatminh2947 on 10/23/19.
//

#include <iostream>
#include <iomanip>
#include "GaussianDataGenerator.h"

int main(int argc, const char *argv[]) {
    double mean = 6;
    double var = 10;

    for (int i = 1; i < argc; i++) {
        std::string para(argv[i]);
        if (para.find("--mean=") == 0) {
            mean = std::stoi(para.substr(para.find('=') + 1));
        } else if (para.find("--var=") == 0) {
            var = std::stoi(para.substr(para.find('=') + 1));
        }
    }

    GaussianDataGenerator gdg(mean, var);

    double est_mean = 0;
    double est_var = 0;
    double M2 = 0;
    int count = 0;

    std::cout << std::setprecision(1) << std::fixed << "Data point source function: N(" << mean << ", " << var << ")"
              << std::endl << std::endl;

    std::cout << std::setprecision(15) << std::fixed;
    for (int i = 0; i < 100000; ++i) {
        double new_data = gdg.generate();

        count++;

        std::cout << "Add new data point: " << new_data << std::endl;

        double prev_delta = new_data - est_mean;
        est_mean = est_mean + prev_delta / count;
        double cur_delta = new_data - est_mean;
        M2 += prev_delta * cur_delta;
        est_var = M2 / count;

        std::cout << "Mean = " << est_mean << "\tVariance = " << est_var << std::endl;
    }
}