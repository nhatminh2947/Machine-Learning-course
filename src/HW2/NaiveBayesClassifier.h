//
// Created by nhatminh2947 on 10/8/19.
//

#ifndef MACHINE_LEARNING_COURSE_NAIVEBAYESCLASSIFIER_H
#define MACHINE_LEARNING_COURSE_NAIVEBAYESCLASSIFIER_H


#include <vector>
#include <cmath>
#include "matrix.h"

class NaiveBayesClassifier {
private:
	int N_ROWS = 28;
	int N_COLS = 28;
	int _mode;
	std::vector<Matrix> preprocess_data(std::vector<Matrix> images);
public:
	explicit NaiveBayesClassifier(int mode=0);
	void train(std::vector<Matrix> train_images, std::vector<int> train_labels);
	void test(std::vector<Matrix> test_images, std::vector<int> test_labels);
};


#endif //MACHINE_LEARNING_COURSE_NAIVEBAYESCLASSIFIER_H
