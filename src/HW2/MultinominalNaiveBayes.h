//
// Created by nhatminh2947 on 10/8/19.
//

#ifndef MACHINE_LEARNING_COURSE_MNISTNAIVEBAYESCLASSIFIER_H
#define MACHINE_LEARNING_COURSE_MNISTNAIVEBAYESCLASSIFIER_H


#include <vector>
#include <matrix.h>
#include "NaiveBayes.h"

class MultinominalNaiveBayes : public NaiveBayes {
private:
	const int N_CLASSES = 10;
	const int N_FEATURES = 28 * 28 * 32;
	Matrix _prop_class = Matrix(N_CLASSES, 1);
	Matrix _weights = Matrix(N_CLASSES, N_FEATURES);
	Matrix _count_feature_class = Matrix(N_CLASSES, 1);
	Matrix _count_class = Matrix(N_CLASSES, 1);
	int _train_size{0};
public:
	MultinominalNaiveBayes();

	void fit(std::vector<Matrix> X, std::vector<int> y) override;

	int predict(Matrix x) override;

	std::vector<double> predict_log_proba(Matrix x, bool normalize) override;
};

std::vector<double> normalize(std::vector<double> v);


#endif //MACHINE_LEARNING_COURSE_MNISTNAIVEBAYESCLASSIFIER_H
