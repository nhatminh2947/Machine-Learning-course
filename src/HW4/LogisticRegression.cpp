//
// Created by nhatminh2947 on 11/3/19.
//

#include <cmath>
#include "LogisticRegression.h"

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}