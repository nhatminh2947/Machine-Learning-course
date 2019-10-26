//
// Created by nhatminh2947 on 9/22/19.
//

#ifndef HOMEWORK_MATRIX_H
#define HOMEWORK_MATRIX_H

#include <iostream>

class Matrix {
private:
    int n, m;
protected:
    double **data;
public:
    Matrix(int n, int m);

    explicit Matrix(int n);

    Matrix() = default;

    Matrix T();

    Matrix flat();

    Matrix power(int x);

    Matrix operator+(Matrix const &b);

    Matrix operator*(Matrix const &b);

    Matrix operator-(Matrix const &b);

    Matrix operator*(double const &lambda);

    Matrix operator/(double const &b);

    friend bool operator==(Matrix const &a, Matrix const &b);

    friend Matrix operator*(double const &lambda, Matrix const &b);

    double &operator()(int i, int j);

    double &operator()(int i, int j) const;

    friend std::ostream &operator<<(std::ostream &out, const Matrix &matrix);

    friend std::ostream &operator<<(std::ostream &out, Matrix &matrix);

    friend double sum(Matrix const &m);

    static double max(Matrix const &m);

    std::pair<Matrix, Matrix> LUDecomposition();

    Matrix inverse();

    int getRows() const;

    int getCols() const;

    static Matrix random(int n, int m);

    static Matrix ToDesignMatrix(double x, int basis);
};

class IdentityMatrix : public Matrix {
private:
    int size;
public:
    explicit IdentityMatrix(int n);

    int getSize() const;
};

class SquareMatrix : public Matrix {
public:
    explicit SquareMatrix(int n);
};

#endif //HOMEWORK_MATRIX_H
