//
// Created by nhatminh2947 on 9/22/19.
//

#ifndef HOMEWORK_MATRIX_H
#define HOMEWORK_MATRIX_H

#include <iostream>
#include <cmath>
#include <random>

#define DEBUG(x) std::cout << #x << " = "<< x << std::endl

template<typename T1, typename T2>
struct is_same_type {
    static const bool value = false;
    static const bool yes = false;
    static const bool no = true;
};


template<typename T1>
struct is_same_type<T1, T1> {
    static const bool value = true;
    static const bool yes = true;
    static const bool no = false;
};


namespace fill {
    struct fill_none {
    };
    struct fill_zeros {
    };
    struct fill_ones {
    };
    struct fill_eye {
    };
    struct fill_random {
    };

    template<typename fill_type>
    struct fill_class {
        inline fill_class() = default;
    };

    static const fill_class<fill_zeros> zeros;
    static const fill_class<fill_ones> ones;
    static const fill_class<fill_eye> eye;
    static const fill_class<fill_random> rand;
}

template<typename Type>
class Matrix {
private:
    int n_{}, m_{};
protected:
    std::vector<std::vector<Type>> data;
//    Type **data;
public:
    Matrix(int n, int m);

//    ~Matrix();

    template<typename fill_type>
    Matrix(int n, int m, const fill::fill_class<fill_type> &fill_class);

    explicit Matrix(int n);

    Matrix<Type>() = default;

    Matrix power(int x);

    Matrix operator+(Matrix const &b);

    Matrix operator-(Matrix const &b);

    Matrix operator*(Matrix const &b);

    Matrix operator*(double const &lambda);

    Matrix operator/(double const &b);

    Matrix row(int i);

    double CalculateDeterminant();

    template<typename T>
    friend Matrix<double> exp(Matrix<T> matrix);

    template<typename fill_type>
    Matrix &fill(const fill::fill_class<fill_type> &f);

    template<typename T>
    friend bool operator==(Matrix<T> const &a, Matrix<T> const &b);

    Type &operator[](int i);

    Type &operator[](int i) const;

    Type &operator()(int i, int j);

    const Type &operator()(int i, int j) const;

    std::pair<Matrix<Type>, Matrix<Type>> LUDecomposition();

    Matrix<Type> inverse();

    int getRows() const;

    int getCols() const;

    const Matrix &zeros();

    const Matrix &ones();

    const Matrix &random();

    const Matrix &eye();

    Matrix<Type> T();

    template<typename T>
    friend std::ostream &operator<<(std::ostream &out, const Matrix<Type> &matrix);

    template<typename T>
    friend std::ostream &operator<<(std::ostream &out, Matrix<Type> &matrix);

    static Matrix<Type> dot(Matrix<Type> a, Matrix<Type> b);
};

template<typename T>
std::ostream &operator<<(std::ostream &out, const Matrix<T> &matrix) {
    for (int i = 0; i < matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            out << matrix(i, j) << " ";
        }
        out << std::endl;
    }

    return out;
}

template<typename Type>
Matrix<Type> Matrix<Type>::operator*(const Matrix<Type> &b) {
    Matrix<Type> result(this->getRows(), b.getCols(), fill::zeros);

    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < b.getCols(); ++j) {
            for (int k = 0; k < this->getCols(); ++k) {
                result.data[i][j] += this->data[i][k] * b(k, j);
            }
        }
    }

    return result;
}

template<typename Type>
Matrix<Type> Matrix<Type>::operator+(const Matrix<Type> &b) {
    Matrix<Type> result(this->n_, this->m_);

    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < m_; ++j) {
            result.data[i][j] = this->data[i][j] + b(i, j);
        }
    }

    return result;
}

template<typename Type>
Type &Matrix<Type>::operator()(int i, int j) {
    if (i >= n_) {
        throw std::out_of_range("Col index is out of range");
    }

    if (j >= m_) {
        throw std::out_of_range("Col index is out of range");
    }

    return this->data[i][j];
}

template<typename Type>
const Type &Matrix<Type>::operator()(int i, int j) const {
    if (i >= n_) {
        throw std::out_of_range("Col index is out of range");
    }

    if (j >= m_) {
        throw std::out_of_range("Col index is out of range");
    }

    return this->data[i][j];
}

template<typename Type>
Matrix<Type> Matrix<Type>::T() {
    Matrix<Type> result(this->getCols(), this->getRows());

    for (int i = 0; i < result.getRows(); ++i) {
        for (int j = 0; j < result.getCols(); ++j) {
            result.data[i][j] = this->data[j][i];
        }
    }

    return result;
}

template<typename Type>
std::pair<Matrix<Type>, Matrix<Type>> Matrix<Type>::LUDecomposition() {
    Matrix<Type> L(n_, n_, fill::zeros);
    Matrix<Type> U(n_, n_, fill::zeros);

    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
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

template<typename Type>
int Matrix<Type>::getRows() const {
    return n_;
}

template<typename Type>
int Matrix<Type>::getCols() const {
    return m_;
}

template<typename Type>
Matrix<Type> Matrix<Type>::operator*(double const &lambda) {
    Matrix<Type> result(this->getRows(), this->getCols());

    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < this->getCols(); ++j) {
            result(i, j) = lambda * this->operator()(i, j);
        }
    }

    return result;
}

template<typename T>
std::ostream &operator<<(std::ostream &out, Matrix<T> &matrix) {
    for (int i = 0; i < matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            out << matrix(i, j) << " ";
        }
        out << std::endl;
    }

    return out;
}

template<typename T>
Matrix<double> operator*(double const &lambda, const Matrix<T> &b) {
    Matrix<double> result(b.getRows(), b.getCols());

    for (int i = 0; i < b.getRows(); ++i) {
        for (int j = 0; j < b.getCols(); ++j) {
            result(i, j) = lambda * b(i, j);
        }
    }

    return result;
}


template<typename Type>
bool operator==(const Matrix<Type> &a, const Matrix<Type> &b) {
    if (a.getRows() == b.getRows() && a.getCols() == b.getCols()) {
        bool equal = true;
        for (int i = 0; i < a.getRows(); ++i) {
            for (int j = 0; j < b.getRows(); ++j) {
                equal &= (std::fabs(a(i, j) - b(i, j)) < 1e-12);
            }
        }

        return equal;
    }

    return false;
}

template<typename Type>
Matrix<Type> Matrix<Type>::inverse() {
    std::pair<Matrix, Matrix> LU = this->LUDecomposition();
    int size = LU.first.getRows();

    Matrix<Type> inverse_L(size, size);
    Matrix<Type> inverse_matrix(size, size);
    Matrix<Type> I(size, size, fill::eye);

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

template<typename Type>
Matrix<Type> Matrix<Type>::operator-(const Matrix<Type> &b) {
    if (this->getRows() != b.getRows() || this->getCols() != b.getCols()) {
        throw std::invalid_argument("Matrix size must be equal");
    }

    Matrix<Type> result = Matrix<Type>(this->getRows(), this->getCols());

    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < this->getCols(); ++j) {
            result(i, j) = this->operator()(i, j) - b(i, j);
        }
    }

    return result;
}


template<typename Type>
Matrix<Type> Matrix<Type>::operator/(double const &b) {
    Matrix<Type> result(this->getRows(), this->getCols());

    for (int i = 0; i < result.getRows(); ++i) {
        for (int j = 0; j < result.getCols(); ++j) {
            result(i, j) = this->operator()(i, j) / b;
        }
    }

    return result;
}

template<typename Type>
Matrix<Type> Matrix<Type>::power(int x) {
    Matrix<Type> result(this->getRows(), this->getCols());

    for (int i = 0; i < result.getRows(); ++i) {
        for (int j = 0; j < result.getCols(); ++j) {
            double value = 1.0;
            for (int k = 0; k < x; ++k) {
                value *= this->operator()(i, j);
            }

            result(i, j) = value;
        }
    }

    return result;
}

template<typename Type>
Matrix<Type>::Matrix(int n, int m) {
    this->n_ = n;
    this->m_ = m;

    for (int i = 0; i < n; ++i) {
        std::vector<Type> row;
        for (int j = 0; j < m; ++j) {
            row.emplace_back(0);
        }

        data.emplace_back(row);
    }

//    data = new Type *[n_];
//    for (size_t i = 0; i < n_; ++i) {
//        data[i] = new Type[m_];
//
//        for (int j = 0; j < m_; ++j) {
//            data[i][j] = 0;
//        }
//    }

//    for (int i = 0; i < n_; ++i) {
//        for (int j = 0; j < m_; ++j) {
//            data[i][j] = 0;
//        }
//    }
}

template<typename Type>
Matrix<Type>::Matrix(int n) {
//    this->n_ = n;
//    this->m_ = n;
//    data = new Type *[n_];
//    for (int i = 0; i < n; ++i) {
//        data[i] = new Type[m_];
//    }
//
//    for (int i = 0; i < n_; ++i) {
//        for (int j = 0; j < m_; ++j) {
//            data[i][j] = 0;
//        }
//    }

    this->n_ = n;
    this->m_ = n;

    for (int i = 0; i < n; ++i) {
        std::vector<Type> row;
        for (int j = 0; j < n; ++j) {
            row.emplace_back(0);
        }

        data.emplace_back(row);
    }
}

template<typename Type>
Type &Matrix<Type>::operator[](int i) {
    return this->operator()(i / this->getCols(), i % this->getCols());
}

template<typename Type>
Type &Matrix<Type>::operator[](int i) const {
    return this->operator()(i / this->getCols(), i % this->getCols());
}

template<typename Type>
Matrix<Type> Matrix<Type>::row(int const i) {
    Matrix<Type> row(1, this->getCols());

    for (int j = 0; j < this->getCols(); ++j) {
        row(0, j) = this->operator()(i, j);
    }

    return row;
}

template<typename Type>
template<typename fill_type>
Matrix<Type> &Matrix<Type>::fill(const fill::fill_class<fill_type> &f) {
    if (is_same_type<fill_type, fill::fill_zeros>::yes) (*this).zeros();
    if (is_same_type<fill_type, fill::fill_ones>::yes) (*this).ones();
    if (is_same_type<fill_type, fill::fill_eye>::yes) (*this).eye();
    if (is_same_type<fill_type, fill::fill_random>::yes) (*this).random();
}

template<typename Type>
const Matrix<Type> &Matrix<Type>::zeros() {
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < m_; ++j) {
            this->operator()(i, j) = 0;
        }
    }

    return *this;
}

template<typename Type>
const Matrix<Type> &Matrix<Type>::ones() {
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < m_; ++j) {
            this->operator()(i, j) = 1;
        }
    }

    return *this;
}

template<typename Type>
const Matrix<Type> &Matrix<Type>::random() {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < m_; ++j) {
            this->operator()(i, j) = distribution(generator);
        }
    }

    return *this;
}

template<typename Type>
const Matrix<Type> &Matrix<Type>::eye() {
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < m_; ++j) {
            if (i == j) this->operator()(i, j) = 1;
            else this->operator()(i, j) = 0;
        }
    }

    return *this;
}

template<typename Type>
template<typename fill_type>
Matrix<Type>::Matrix(int n, int m, const fill::fill_class<fill_type> &fill_class) : Matrix(n, m) {
    (*this).fill(fill_class);
}

template<typename T>
Matrix<double> exp(Matrix<T> matrix) {
    for (int i = 0; i < matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            matrix(i, j) = exp(matrix(i, j));
        }
    }
    return matrix;
}

template<typename Type>
double Matrix<Type>::CalculateDeterminant() {
    std::pair<Matrix, Matrix> LU = LUDecomposition();

    double det_L = 1;
    double det_U = 1;

    for (int i = 0; i < LU.first.getRows(); ++i) {
        det_L *= LU.first(i, i);
    }

    for (int i = 0; i < LU.first.getRows(); ++i) {
        det_U *= LU.second(i, i);
    }

    return det_L * det_U;
}

template<typename Type>
Matrix<Type> Matrix<Type>::dot(Matrix<Type> a, Matrix<Type> b) {
    Matrix<Type> result(a.getRows(), a.getCols());

    for (int i = 0; i < a.getRows(); ++i) {
        for (int j = 0; j < a.getCols(); ++j) {
            result(i, j) = a(i, j) * b(i, j);
        }
    }

    return result;
}

//template<typename Type>
//Matrix<Type>::~Matrix() {
//    for (int i = 0; i < n_; ++i) {
//        delete[] data[i];
//    }
//    delete[] data;
//    data = nullptr;
//}

#endif //HOMEWORK_MATRIX_H
