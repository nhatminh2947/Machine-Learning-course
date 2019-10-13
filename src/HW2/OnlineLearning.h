//
// Created by nhatminh2947 on 10/13/19.
//

#ifndef HOMEWORK_ONLINELEARNING_H
#define HOMEWORK_ONLINELEARNING_H


#include <string>

class OnlineLearning {
private:
	int _a;
	int _b;

	static double P(int x, double lambda);

public:
	OnlineLearning(int a, int b);

	double Event(const std::string &event);

	double factorial(int x);

	int getA() const;

	int getB() const;
};


#endif //HOMEWORK_ONLINELEARNING_H
