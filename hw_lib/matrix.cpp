//
// Created by nhatminh2947 on 9/22/19.
//

#include <cmath>
#include <random>
#include "matrix.h"

Matrix::Matrix(int n, int m) {
    this->n = n;
    this->m = m;
    data = new double *[n];
    for (int i = 0; i < n; ++i) {
        data[i] = new double[m];
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            data[i][j] = 0;
        }
    }
}

Matrix::Matrix(int n) {
    this->n = n;
    this->m = n;
    data = new double *[n];
    for (int i = 0; i < n; ++i) {
        data[i] = new double[m];
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            data[i][j] = 0;
        }
    }
}

Matrix Matrix::operator+(Matrix const &b) {
    Matrix result(this->n, this->m);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result.data[i][j] = this->data[i][j] + b(i, j);
        }
    }

    return result;
}

Matrix Matrix::operator*(Matrix const &b) {
    Matrix result(this->getRows(), b.getCols());

    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < b.getCols(); ++j) {
            for (int k = 0; k < m; ++k) {
                result.data[i][j] += this->data[i][k] * b(k, j);
            }
        }
    }

    return result;
}

double &Matrix::operator()(int i, int j) {
    if (i >= n) {
        throw std::out_of_range("Row index is out of range");
    }

    if (j >= m) {
        throw std::out_of_range("Col index is out of range");
    }

    return this->data[i][j];
}

double &Matrix::operator()(int i, int j) const {
    if (i >= n) {
        throw std::out_of_range("Row index is out of range");
    }

    if (j >= m) {
        throw std::out_of_range("Col index is out of range");
    }

    return this->data[i][j];
}

Matrix Matrix::T() {
    Matrix result(this->getCols(), this->getRows());

    for (int i = 0; i < result.getRows(); ++i) {
        for (int j = 0; j < result.getCols(); ++j) {
            result.data[i][j] = this->data[j][i];
        }
    }

    return result;
}

IdentityMatrix::IdentityMatrix(int n) : Matrix(n) {
    this->size = n;
    for (int i = 0; i < n; ++i) {
        this->data[i][i] = 1;
    }
}

int IdentityMatrix::getSize() const {
    return size;
}

std::ostream &operator<<(std::ostream &out, const Matrix &matrix) {
    for (int i = 0; i < matrix.n; ++i) {
        for (int j = 0; j < matrix.m; ++j) {
            out << matrix(i, j) << " ";
        }
        out << std::endl;
    }

    return out;
}

std::ostream &operator<<(std::ostream &out, Matrix &matrix) {
    for (int i = 0; i < matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            out << matrix(i, j) << " ";
        }
        out << std::endl;
    }

    return out;
}

std::pair<Matrix, Matrix> Matrix::LUDecomposition() {
    Matrix L(n);
    Matrix U(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            if (i <= j) {
                for (int k = 0; k < i; ++k) {
                    sum += U(k, j) * L(i, k);
                }
                U.data[i][j] = this->data[i][j] - sum;
            }

            if (i >= j) {
                sum = 0;
                for (int k = 0; k < j; ++k) {
                    sum += U(k, j) * L(i, k);
                }

                L.data[i][j] = (this->data[i][j] - sum) / U(j, j);
            }
        }
    }

    return std::make_pair(L, U);
}

int Matrix::getRows() const {
    return n;
}

int Matrix::getCols() const {
    return m;
}

Matrix Matrix::operator*(const double &lambda) {
    Matrix result(this->getRows(), this->getCols());

    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < this->getCols(); ++j) {
            result(i, j) = lambda * this->operator()(i, j);
        }
    }

    return result;
}

Matrix operator*(double const &a, Matrix const &b) {
    Matrix result(b.getRows(), b.getCols());

    for (int i = 0; i < b.getRows(); ++i) {
        for (int j = 0; j < b.getCols(); ++j) {
            result(i, j) = a * b(i, j);
        }
    }

    return result;
}

bool operator==(Matrix const &a, Matrix const &b) {
    if (a.getRows() == b.getRows() && a.getCols() == b.getCols()) {
        bool equal = true;
        for (int i = 0; i < a.getRows(); ++i) {
            for (int j = 0; j < b.getRows(); ++j) {
                equal &= (fabs(a(i, j) - b(i, j)) < 1e-12);
            }
        }

        return equal;
    }

    return false;
}

Matrix Matrix::inverse() {
    std::pair<Matrix, Matrix> LU = this->LUDecomposition();
    int size = LU.first.getRows();

    Matrix inverse_L = Matrix(size, size);
    Matrix inverse_matrix = Matrix(size, size);
    IdentityMatrix I = IdentityMatrix(size);

    for (int k = 0; k < size; ++k) {
        for (int i = 0; i < size; ++i) {
            double sum = 0.0;
            for (int j = 0; j < i; ++j) {
                sum += LU.first(i, j) * inverse_L(j, k);
            }
            inverse_L(i, k) = (I(i, k) - sum) / LU.first(i, i);
        }
    }

    for (int k = 0; k < size; ++k) {
        for (int i = size - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = size - 1; j > i; --j) {
                sum += LU.second(i, j) * inverse_matrix(j, k);
            }

            inverse_matrix(i, k) = (inverse_L(i, k) - sum) / LU.second(i, i);
        }
    }

    return inverse_matrix;
}

Matrix Matrix::operator-(Matrix const &b) {
    if (this->getRows() != b.getRows() || this->getCols() != b.getCols()) {
        throw std::invalid_argument("Matrix size must be equal");
    }

    Matrix result = Matrix(this->getRows(), this->getCols());

    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < this->getCols(); ++j) {
            result(i, j) = this->operator()(i, j) - b(i, j);
        }
    }

    return result;
}

double sum(Matrix const &m) {
    double total = 0;
    for (int i = 0; i < m.getRows(); ++i) {
        for (int j = 0; j < m.getCols(); ++j) {
            total += m(i, j);
        }
    }

    return total;
}

Matrix Matrix::random(int n, int m) {
    Matrix rand = Matrix(n, m);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            rand(i, j) = dis(gen);
        }
    }

    return rand;
}

double Matrix::max(Matrix const &m) {
    double result = INT64_MIN;
    for (int i = 0; i < m.getRows(); ++i) {
        for (int j = 0; j < m.getCols(); ++j) {
            result = std::max(m(i, j), result);
        }
    }

    return result;
}

Matrix Matrix::operator/(double const &b) {
    Matrix result(this->getRows(), this->getCols());

    for (int i = 0; i < result.getRows(); ++i) {
        for (int j = 0; j < result.getCols(); ++j) {
            result(i, j) = this->operator()(i, j) / b;
        }
    }

    return result;
}

Matrix Matrix::flat() {
    Matrix result(1, this->getCols() * this->getRows());

    for (int i = 0; i < this->getCols() * this->getRows(); ++i) {
        int r = i / this->getCols();
        int c = i % this->getCols();
        result(0, i) = this->operator()(r, c);
    }

    return result;
}

Matrix Matrix::power(int x) {
    Matrix result( this->getRows(),this->getCols());

    for (int i = 0; i < result.getRows(); ++i) {
        for (int j = 0; j < result.getCols(); ++j) {
            double value = 1.0;
            for (int k = 0; k < x; ++k) {
                value *= this->operator()(i, j);
            }

            result(i, j) = value;
        }
    }

    return result;}

Matrix Matrix::ToDesignMatrix(double x, int basis) {
    Matrix X(basis, 1);
    for (int i = 0; i < basis; ++i) {
        X(i, 0) = pow(x, i);
    }

    return X;
}

SquareMatrix::SquareMatrix(int n) : Matrix(n) {

}
