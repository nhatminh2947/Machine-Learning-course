#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <limits>
#include "matrix.h"
#include "Dataset.h"

Matrix first_order_f(Matrix X, Matrix Y, Matrix W) {
	return W.T() * X.T() * X - Y.T() * X;
}

Matrix second_order_f(Matrix X) {
	return X.T() * X;
}

int main(int argc, char *argv[]) {
	double lambda = 0.5;
	int bases = 2;
	std::string file_input;
	std::string file_output = "./src/HW1/coeffs.txt";

	for (int i = 1; i < argc; i++) {
		std::string para(argv[i]);
		if (para.find("--lambda=") == 0) {
			lambda = std::stod(para.substr(para.find("=") + 1));
		} else if (para.find("--bases=") == 0) {
			bases = std::stoi(para.substr(para.find("=") + 1));
		} else if (para.find("--input=") == 0) {
			file_input = para.substr(para.find("=") + 1);
		} else if (para.find("--output=") == 0) {
			file_output = para.substr(para.find("=") + 1);
		}
	}

	Dataset dataset;

	std::ifstream input_file(file_input);

	std::string line;
	if (input_file.is_open()) {
		double x, y;
		while (getline(input_file, line)) {
			char c;
			std::istringstream iss(line);
			iss >> x >> c >> y;
            dataset.Add(Point(x, y));
		}
		input_file.close();
	} else {
		std::cout << "File is closed" << std::endl;
	}

	Matrix X = dataset.ToDesignMatrix(bases);
	Matrix Y = dataset.GetLabels();

	IdentityMatrix I = IdentityMatrix(X.getRows());

	Matrix W = Matrix(bases, 1);

	for (int i = 0; i < 10; ++i) {
//		std::cout << W << std::endl;
		std::cout << "Gradient:\n" << first_order_f(X, Y, W) << std::endl;
		std::cout << "Hessian:\n" << second_order_f(X) << std::endl;

		W = W - (first_order_f(X, Y, W) * second_order_f(X).inverse()).T();

	}

	std::cout << "Newton's Method:" << std::endl;
	std::cout << "Fitting line: ";

	std::ofstream out(file_output);
	if (out.is_open()) {
		for (int i = 0; i < W.getRows(); ++i) {
			out << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << W(i, 0) << std::endl;
			std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << W(W.getRows() - i - 1, 0);

			if (W.getRows() - i - 1 > 0) {
				std::cout << "X^" << W.getRows() - i - 1 << " + ";
			} else {
				std::cout << std::endl;
			}
		}
		out.close();
	} else {
		std::cout << "File output is closed" << std::endl;
	}

	double total_error = sum((X * W - Y).T() * (X * W - Y));
	std::cout << "Total error: " << total_error << std::endl;

	return 0;
}