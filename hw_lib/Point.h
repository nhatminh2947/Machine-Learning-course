//
// Created by nhatminh2947 on 9/24/19.
//

#ifndef HOMEWORK_POINT_H
#define HOMEWORK_POINT_H

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include "matrix.h"


template<typename T = double, unsigned int n_dimensions = 2>
class Point {
private:
    std::array<T, n_dimensions> elements_;

public:
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

//    Col<T> ToCol() {
//        Col<T> col(elements_.size());
//
//        for (int i = 0; i < col.size(); ++i) {
//            col(i) = elements_.at(i);
//        }
//
//        return col;
//    }
};

#endif //HOMEWORK_POINT_H
