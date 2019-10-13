//
// Created by nhatminh2947 on 10/8/19.
//

#ifndef MACHINE_LEARNING_COURSE_MNISTNAIVEBAYESCLASSIFIER_H
#define MACHINE_LEARNING_COURSE_MNISTNAIVEBAYESCLASSIFIER_H


#include <vector>
#include <matrix.h>
#include "NaiveBayes.h"

class MultinomialNaiveBayes : public NaiveBayes {
private:
	const int N_CLASSES = 10;
	const int N_FEATURES = N_ROWS * N_COLS * 32;
	Matrix _prior = Matrix(N_CLASSES, 1);
	Matrix _weights = Matrix(N_CLASSES, N_FEATURES);
	Matrix _count_feature_class = Matrix(N_CLASSES, 1);
	int _train_size{0};
public:
	MultinomialNaiveBayes();

	void fit(std::vector<Matrix> X, std::vector<int> y) override;

	int predict(Matrix x) override;

	std::vector<double> predict_log_proba(Matrix x, bool normalize) override;

	void imagination() override ;
};


#endif //MACHINE_LEARNING_COURSE_MNISTNAIVEBAYESCLASSIFIER_H
