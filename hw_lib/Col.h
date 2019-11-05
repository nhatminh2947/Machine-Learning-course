//
// Created by nhatminh2947 on 11/5/19.
//

#ifndef HOMEWORK_COL_H
#define HOMEWORK_COL_H


#include "matrix.h"

template<typename Type>
class Col : public Matrix<Type> {
public:
    explicit Col(int n);

    template<typename fill_type>
    Col(int n, const fill::fill_class<fill_type> &f);

    Col(Matrix<Type> matrix);

    template<typename T>
    friend Matrix<double> operator*(Matrix<double> a, Col<T> b);

    Col &operator=(Matrix<Type> const &b);

    int size();
};

//template<typename Type>
//Col<Type> &Col<Type>::operator=(Col<Type> const &b) {
//    for (int i = 0; i < this->getRows(); ++i) {
//        this->operator[](i) = b[i];
//    }
//}
//
template<typename Type>
Col<Type> &Col<Type>::operator=(Matrix<Type> const &b) {
    for (int i = 0; i < this->getRows(); ++i) {
        this->operator[](i) = b(i, 0);
    }
}

template<typename Type>
Col<Type>::Col(int n):Matrix<Type>(n, 1) {

}

//template<typename Type>
//Col<Type> Col<Type>::operator-(const Col<Type> &b) {
//    if (this->getRows() != b.getRows()) {
//        throw std::invalid_argument("Matrix size must be equal");
//    }
//
//    Col<Type> result = Col<Type>(this->getRows());
//
//    for (int i = 0; i < this->getRows(); ++i) {
//        result[i] = this->operator[](i) - b[i];
//    }
//
//    return result;
//}



template<typename T>
int Col<T>::size() {
    return this->getRows();
}

template<typename Type>
Col<Type>::Col(Matrix<Type> matrix) : Matrix<Type>(matrix.getRows(), 1) {
    if (matrix.getCols() != 1) {
        throw std::invalid_argument("Matrix must be in shape (n, 1)");
    }

    for (int i = 0; i < this->getRows(); ++i) {
        this->operator[](i) = matrix(i, 0);
    }
}

template<typename Type>
template<typename fill_type>
Col<Type>::Col(int n, const fill::fill_class<fill_type> &f):Matrix<Type>(n, 1, f) {

}

template<typename T>
Matrix<double> operator*(Matrix<double> a, Col<T> b) {
    Matrix<double> result(a.getRows(), 1);

    for (int i = 0; i < a.getRows(); ++i) {
        for (int k = 0; k < a.getRows(); ++k) {
            result(i, 0) += a(i, k) * b[k];
        }
    }

    return result;
}

#endif //HOMEWORK_COL_H
