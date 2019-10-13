//
// Created by nhatminh2947 on 10/13/19.
//

#include <cmath>
#include "OnlineLearning.h"

OnlineLearning::OnlineLearning(int a, int b) {
	_a = a;
	_b = b;
}

double OnlineLearning::Event(const std::string &event) {
	int n = 0;
	int N = event.size();
	for (char e : event) {
		if (e == '1') {
			n++;
		}
	}

	double likelihood = factorial(N) / (factorial(n) * factorial(N - n)) * pow((1.0 * n / N), n) * pow(1 - (1.0 * n / N), N - n);

	_a += n;
	_b += (N - n);

	return likelihood;
}


int OnlineLearning::getA() const {
	return _a;
}

int OnlineLearning::getB() const {
	return _b;
}

double OnlineLearning::P(int x, double lambda) {
	return std::pow(lambda, x) * std::pow(1 - lambda, x);
}

double OnlineLearning::factorial(int x) {
	double result = 1;
	while (x) {
		result *= x;
		x--;
	}

	return result;
}
