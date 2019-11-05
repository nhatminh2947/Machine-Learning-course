//
// Created by nhatminh2947 on 11/3/19.
//

#ifndef HOMEWORK_LOGISTICREGRESSION_H
#define HOMEWORK_LOGISTICREGRESSION_H


#include <Point.h>
#include <Col.h>

template<typename OptimizerType>
class LogisticRegression {
private:
    OptimizerType optimizer_;
    Col<double> weights_;
    double learning_rate_;
    int max_iter_;

    static double sigmoid(double const &x) {
        return 1.0 / (1.0 + exp(-x));
    }

    static Col<double> sigmoid(Col<double> x) {
        Col<double> sigmoid_x(x.size());

        for (int i = 0; i < x.size(); ++i) {
            sigmoid_x[i] = sigmoid(x[i]);
        }

        return sigmoid_x;
    }

    static Col<int> ToClass(Col<double> x) {
        Col<int> result(x.size());

        for (int i = 0; i < x.size(); ++i) {
            result[i] = (x[i] > 0.5);
        }

        return result;
    }

    double ErrorFunction() {

    }

    void Train(Matrix<double> X, Col<int> y) {
        NewtonMethod(X, y);
    }

    void GradientDescent(Matrix<double> X, Col<int> y) {
        for (int i = 0; i < max_iter_; ++i) {
            for (int j = 0; j < X.getRows(); ++j) {
                Col<double> z = weights_.T() * Col<double>(X.row(j).T());
                Col<double> tmp = (y[j] - sigmoid(z[0])) * Col<double>(X.row(j).T());
                weights_ = weights_ + learning_rate_ * tmp;
            }

            std::cout << "weights" << std::endl;
            std::cout << weights_ << std::endl;
        }

        std::cout << "Training Completed" << std::endl;
        std::cout << weights_ << std::endl;
    }

    void NewtonMethod(Matrix<double> X, Col<int> y) {
        for (int i = 0; i < max_iter_; ++i) {
            Matrix<double> H = X.T() * X;
            std::cout << "H" << std::endl;
            std::cout << H << std::endl;
            std::cout << "H.inverse()" << std::endl;
            std::cout << H.inverse() << std::endl;
            Matrix<double> gradient = X.T() * X * weights_ - X.T() * y;
            std::cout << "gradient" << std::endl;
            std::cout << gradient << std::endl;
            std::cout << "H.inverse() * gradient" << std::endl;
            std::cout << H.inverse() * gradient << std::endl;
            weights_ = weights_ - H.inverse() * gradient;
        }

        std::cout << "Training Completed" << std::endl;
        std::cout << weights_ << std::endl;
    }

public:
    explicit LogisticRegression(const Matrix<double> &X, const Col<int> &y, double learning_rate, int max_iter)
            : weights_(X.getCols(), fill::rand),
              learning_rate_(learning_rate),
              max_iter_(max_iter) {
        Train(X, y);
    }

    Col<int> Classify(Matrix<double> X) {
        Col<double> z = Col<double>(X * weights_);
        z = sigmoid(z);

        return ToClass(z);
    }
};

#endif //HOMEWORK_LOGISTICREGRESSION_H
