#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <limits>
#include "matrix.h"
#include "Point.h"

int main(int argc, char *argv[]) {
	double lambda = 0;
	int bases = 2;
	std::string file_input;
	std::string file_output = "../src/HW1/coeffs.txt";

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
	Matrix y = dataset.GetLabels();

	IdentityMatrix I = IdentityMatrix(X.getRows());

	Matrix coeffs = (X.T() * X + lambda * I).inverse() * X.T() * y;
	std::cout << (X.T() * X + lambda * I).inverse() << std::endl;
	std::cout << coeffs << std::endl;

	std::cout << "LSE:" << std::endl;
	std::cout << "Fitting line: ";

	std::ofstream out(file_output);
	if (out.is_open()) {
		for (int i = 0; i < coeffs.getRows(); ++i) {
			out << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << coeffs(i, 0) << std::endl;
			std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << coeffs(coeffs.getRows() - i - 1, 0);

			if (coeffs.getRows() - i - 1 > 0) {
				std::cout << "X^" << coeffs.getRows() - i - 1 << " + ";
			} else {
				std::cout << std::endl;
			}
		}
		out.close();
	} else {
		std::cout << "File output is closed" << std::endl;
	}

	double total_error = sum((X * coeffs - y).T() * (X * coeffs - y));
	std::cout << "Total error: " << total_error << std::endl;

	return 0;
}