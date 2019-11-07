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

    static Matrix<double> sigmoid(Matrix<double> x) {
        Matrix<double> sigmoid_x(x.getRows(), x.getCols());

        for (int i = 0; i < sigmoid_x.getRows(); ++i) {
            for (int j = 0; j < sigmoid_x.getCols(); ++j) {
                sigmoid_x(i, j) = sigmoid(x(i, j));
            }
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

    void Train(Matrix<double> X, Col<double> y) {
//        GradientDescent(X, y);
        NewtonMethod(X, y);
    }

    Matrix<double> CalculateGradient(Matrix<double> X, Col<double> y) {
        Matrix<double> s = X * weights_;
        Matrix<double> sigmoid_Xw = sigmoid(s);
        return X.T() * (y - sigmoid_Xw);
    }

    void GradientDescent(Matrix<double> X, Col<double> y) {
        for (int i = 0; i < max_iter_; ++i) {
            weights_ = weights_ + learning_rate_ * CalculateGradient(X, y);

//            std::cout << "weights" << std::endl;
//            std::cout << weights_ << std::endl;
        }

//        std::cout << "Training Completed" << std::endl;
//        std::cout << weights_ << std::endl;
    }

    Matrix<double> CalculateHessian(Matrix<double> X) {
        Matrix<double> D(X.getRows(), X.getRows(), fill::eye);

        for (int i = 0; i < X.getRows(); ++i) {
            double s = (X.row(i) * weights_)[0];
            D(i, i) = sigmoid(s) * (1 - sigmoid(s));
        }

        return X.T() * D * X;
    }

    void NewtonMethod(Matrix<double> X, Col<double> y) {
        for (int i = 0; i < max_iter_; ++i) {
            Matrix<double> H = CalculateHessian(X);
            Matrix<double> gradient = CalculateGradient(X, y);
            DEBUG(H);
            DEBUG(H.CalculateDeterminant());
            double det_H = H.CalculateDeterminant();
            if (std::isnan(det_H) || det_H == 0) {
                std::cout << "Using steepest descent" << std::endl;
                weights_ = weights_ + learning_rate_ * gradient;
            } else {
                std::cout << "Using newton method" << std::endl;
                weights_ = weights_ + H.inverse() * gradient;
            }

        }

        std::cout << "Training Completed" << std::endl;
        std::cout << weights_ << std::endl;
    }

public:
    explicit LogisticRegression(const Matrix<double> &X, const Col<double> &y, double learning_rate, int max_iter)
            : weights_(X.getCols(), fill::zeros),
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
