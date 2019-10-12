//
// Created by nhatminh2947 on 10/10/19.
//

#include "NaiveBayes.h"

std::vector<double> NaiveBayes::normalize(std::vector<double> v) {
	double sum = 0;
	for (double i : v) {
		sum += i;
	}

	for (double &i : v) {
		i /= sum;
	}

	return v;
}
