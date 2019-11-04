//
// Created by nhatminh2947 on 9/24/19.
//

#ifndef HOMEWORK_DATASET_H
#define HOMEWORK_DATASET_H

#include <iostream>
#include <vector>
#include <array>
#include "matrix.h"


template<typename T, unsigned int n_dimensions = 2>
class Point {
private:
    std::array<T, n_dimensions> elements_;

public:
    typedef T ValueType;

    T &operator[](int const i) {
        return elements_[i];
    }

    T const &operator[](int const i) const {
        return elements_[i];
    }

    void operator+=(Point const &other) {
        for (int i = 0; i < n_dimensions; ++i) {
            elements_[i] += other.elements_[i];
        }
    }

    void operator-=(Point const &other) {
        for (int i = 0; i < n_dimensions; ++i) {
            elements_[i] -= other.elements_[i];
        }
    }

    friend Point operator+(Point const &a, Point const &b) {
        Point ret(a);

        ret += b;
        return ret;
    }

    friend Point operator-(Point const &a, Point const &b) {
        Point ret(a);

        ret -= b;
        return ret;
    }

    Point() : elements_() {}

    template<typename ... Args>
    explicit Point(const Args &... args) : elements_{args...} {}

    Point(T x, T y) {
        elements_[0] = x;
        elements_[1] = y;
    }

    Point(T x, T y, T z) {
        elements_[0] = x;
        elements_[1] = y;
        elements_[2] = z;
    }
};

template<unsigned int nDimensions = 2>
class Dataset {
private:
    std::vector<Point<double, nDimensions>> points_;
    std::vector<int> labels_;

public:
    void Add(Point<double, nDimensions> point, int label);

    Matrix GetPoints();

    Matrix GetLabels();

    int size();
};

template<unsigned int nDimensions>
void Dataset<nDimensions>::Add(Point<double, nDimensions> point, int label) {
    points_.emplace_back(point);
    labels_.emplace_back(label);
}

template<unsigned int nDimensions>
Matrix Dataset<nDimensions>::GetPoints() {
    Matrix result(points_.size(), nDimensions + 1);

    for (int i = 0; i < points_.size(); ++i) {
        result(i, 0) = 1;
        for (int j = 0; j < nDimensions; ++j) {
            result(i, j + 1) = points_[i][j];
        }
    }

    return result;
}

template<unsigned int nDimensions>
Matrix Dataset<nDimensions>::GetLabels() {
    Matrix lables(labels_.size(), 1);

    for (int i = 0; i < labels_.size(); ++i) {
        lables(i, 0) = labels_[i];
    }

    return lables;
}

template<unsigned int nDimensions>
int Dataset<nDimensions>::size() {
    return labels_.size();
}

#endif //HOMEWORK_DATASET_H
