//
// Created by nhatminh2947 on 9/24/19.
//

#ifndef HOMEWORK_DATASET_H
#define HOMEWORK_DATASET_H

#include <iostream>
#include <vector>
#include "matrix.h"

class Point {
public:
	Point(double x, double y);

	double getX() const;

	double getY() const;

private:
	double _x, _y;

};

class Dataset {
private:
	std::vector<Point> data;
public:
	void add(Point point);

	Matrix toDesignMatrix(int bases = 2);

	Matrix labels();
};


#endif //HOMEWORK_DATASET_H
