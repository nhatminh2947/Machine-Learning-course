//
// Created by nhatminh2947 on 9/22/19.
//

#ifndef HOMEWORK_MATRIX_H
#define HOMEWORK_MATRIX_H

#include <iostream>

template<typename Type = double>
class Matrix {
private:
    int n_{}, m_{};
protected:
    Type **data;
public:
    Matrix(int n, int m);

    explicit Matrix(int n);

    Matrix<Type>() = default;

    Matrix<Type> T();

    Matrix<Type> power(int x);

    Matrix<Type> operator+(Matrix<Type> const &b);

    Matrix<Type> operator*(Matrix<Type> const &b);

    Matrix<Type> operator-(Matrix<Type> const &b);

    Matrix<Type> operator*(double const &lambda);

    Matrix<Type> operator/(double const &b);

    template<typename T>
    friend bool operator==(Matrix<T> const &a, Matrix<T> const &b);

    template<typename T>
    friend Matrix<Type> operator*(double const &lambda, Matrix<T> const &b);

    Type &operator()(int i, int j);

    Type &operator()(int i, int j) const;

    template<typename T>
    friend std::ostream &operator<<(std::ostream &out, const Matrix<T> &matrix);

    template<typename T>
    friend std::ostream &operator<<(std::ostream &out, Matrix<T> &matrix);

    std::pair<Matrix<Type>, Matrix<Type>> LUDecomposition();

    Matrix<Type> inverse();

    int getRows() const;

    int getCols() const;

    Matrix<Type> eye(int n);
};

template<typename T = int>
class Row : public Matrix<T> {
public:
    explicit Row(int n);

    T &operator()(int i);

    T &operator()(int i) const;

    int size();
};

template<typename T>
T &Row<T>::operator()(int i) {
    return Matrix<T>::operator()(i, 0);
}

template<typename T>
T &Row<T>::operator()(int i) const {
    return Matrix<T>::operator()(i, 0);
}

template<typename T>
Row<T>::Row(int n) : Matrix<T>(n, 1) {

}

template<typename T>
int Row<T>::size() {
    return this->getRows();
}


template<typename Type>
std::ostream &operator<<(std::ostream &out, const Matrix<Type> &matrix) {
    for (int i = 0; i < matrix.n_; ++i) {
        for (int j = 0; j < matrix.m_; ++j) {
            out << matrix(i, j) << " ";
        }
        out << std::endl;
    }

    return out;
}

template<typename Type>
Matrix<Type> Matrix<Type>::operator*(const Matrix<Type> &b) {
    Matrix<Type> result(this->getRows(), b.getCols());

    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < b.getCols(); ++j) {
            for (int k = 0; k < m_; ++k) {
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
        throw std::out_of_range("Row index is out of range");
    }

    if (j >= m_) {
        throw std::out_of_range("Col index is out of range");
    }

    return this->data[i][j];
}

template<typename Type>
Type &Matrix<Type>::operator()(int i, int j) const {
    if (i >= n_) {
        throw std::out_of_range("Row index is out of range");
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
    Matrix<Type> L(n_);
    Matrix<Type> U(n_);

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

template<typename Type>
std::ostream &operator<<(std::ostream &out, Matrix<Type> &matrix) {
    for (int i = 0; i < matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            out << matrix(i, j) << " ";
        }
        out << std::endl;
    }

    return out;
}

template<typename Type>
Matrix<Type> operator*(double const &lambda, const Matrix<Type> &b) {
    Matrix<Type> result(b.getRows(), b.getCols());

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
                equal &= (fabs(a(i, j) - b(i, j)) < 1e-12);
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

    Matrix inverse_L = Matrix(size, size);
    Matrix inverse_matrix = Matrix(size, size);
    Matrix<Type> I = Matrix<Type>(size).eye(size);

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
Matrix<Type> Matrix<Type>::eye(int n) {
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < m_; ++j) {
            if (i == j) this->operator()(i, j) = 1;
            else this->operator()(i, j) = 0;
        }
    }
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
    data = new Type *[n_];
    for (int i = 0; i < n_; ++i) {
        data[i] = new Type[m_];
    }

    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < m_; ++j) {
            data[i][j] = 0;
        }
    }
}

template<typename Type>
Matrix<Type>::Matrix(int n) {
    this->n_ = n;
    this->m_ = n;
    data = new Type *[n_];
    for (int i = 0; i < n; ++i) {
        data[i] = new Type[m_];
    }

    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < m_; ++j) {
            data[i][j] = 0;
        }
    }
}


#endif //HOMEWORK_MATRIX_H
