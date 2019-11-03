//
// Created by nhatminh2947 on 9/24/19.
//

#include "Dataset.h"

Point::Point(double x, double y) {
	this->_x = x;
	this->_y = y;
}

double Point::getX() const {
	return _x;
}

double Point::getY() const {
	return _y;
}

void Dataset::Add(Point point) {
	data.emplace_back(point);
}

Matrix Dataset::ToDesignMatrix(int bases) {
	Matrix matrix_data = Matrix(this->data.size(), bases);

	for (int i = 0; i < matrix_data.getRows(); ++i) {
		double base = this->data[i].getX();
		double x = 1;
		for (int j = 0; j < matrix_data.getCols(); ++j) {
			matrix_data(i, j) = x;
			x *= base;
		}
	}

	return matrix_data;
}

Matrix Dataset::GetLabels() {
	Matrix y = Matrix(this->data.size(), 1);

	for (int i = 0; i < y.getRows(); ++i) {
		y(i, 0) = this->data[i].getY();
	}

	return y;
}
