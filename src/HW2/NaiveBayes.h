//
// Created by nhatminh2947 on 10/10/19.
//

#ifndef MACHINE_LEARNING_COURSE_NAIVEBAYES_H
#define MACHINE_LEARNING_COURSE_NAIVEBAYES_H


#include <matrix.h>
#include <vector>

class NaiveBayes {
protected:
    const int N_CLASSES = 10;
	const int N_ROWS = 28;
	const int N_COLS = 28;
    int _train_size={};
public:
    virtual void fit(std::vector<Matrix> X, std::vector<int> y) = 0;
    virtual int predict(Matrix x) = 0;
    virtual std::vector<double> predict_log_proba(Matrix x, bool normalize) = 0;
	std::vector<double> normalize(std::vector<double> v);
};


#endif //MACHINE_LEARNING_COURSE_NAIVEBAYES_H
