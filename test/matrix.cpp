//
// Created by nhatminh2947 on 9/23/19.
//

#include "gtest/gtest.h"
#include "matrix.h"

class MatrixInitFixture : public ::testing::Test {
public:
	MatrixInitFixture() = default;

	Matrix a = Matrix(3, 3);
	Matrix b = Matrix(3, 3);
	IdentityMatrix I = IdentityMatrix(5);
};

TEST_F(MatrixInitFixture, InitMatrixAllZero) {
	for (int i = 0; i < a.getRows(); ++i) {
		for (int j = 0; j < a.getCols(); ++j) {
			EXPECT_EQ(a(i, j), 0);
		}
	}
}

TEST_F(MatrixInitFixture, InitIdentityMatrix) {
	for (int i = 0; i < I.getSize(); ++i) {
		for (int j = 0; j < I.getSize(); ++j) {
			if (i != j) {
				EXPECT_EQ(I(i, j), 0);
			} else {
				EXPECT_EQ(I(i, j), 1);

			}
		}
	}
}

TEST(MatrixOperatorTestSuit, MatrixOperatorTestSuit_Assign_Test) {
	Matrix a = Matrix(3, 3);
	a(0, 0) = 1;
	a(2, 2) = 3;
	a(1, 2) = 9;

	EXPECT_EQ(a(0, 0), 1);
	EXPECT_EQ(a(2, 2), 3);
	EXPECT_EQ(a(1, 2), 9);
}

TEST(MatrixOperatorTestSuit, MatrixOperatorTestSuit_Add_Test) {
	Matrix a = Matrix(3, 3);
	Matrix b = Matrix(3, 3);

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			a(i, j) = 1;
			b(i, j) = 2;
		}
	}

	Matrix c = a + b;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			EXPECT_EQ(c(i, j), 3);
		}
	}
}

TEST(MatrixOperatorTestSuit, MatrixOperatorTestSuit_Equal_Test) {
	Matrix a = Matrix(3, 3);
	Matrix b = Matrix(2, 3);
	Matrix c = Matrix(2, 3);

	EXPECT_FALSE(a == b);

	b(0, 0) = 6;
	b(0, 1) = 3;
	b(0, 2) = 5;
	b(1, 0) = 3;
	b(1, 1) = 9;
	b(1, 2) = 7;

	c(0, 0) = 6;
	c(0, 1) = 3;
	c(0, 2) = 5;
	c(1, 0) = 3;
	c(1, 1) = 9;
	c(1, 2) = 7;

	EXPECT_TRUE(c == b);
}

TEST(MatrixOperatorTestSuit, MatrixOperatorTestSuit_Multiplication_Test) {
	Matrix a = Matrix(3, 3);
	Matrix b = Matrix(3, 3);

	a(0, 0) = 4;
	a(0, 1) = 3;
	a(0, 2) = 7;
	a(1, 0) = 9;
	a(1, 1) = 2;
	a(1, 2) = 3;
	a(2, 0) = 2;
	a(2, 1) = 1;
	a(2, 2) = 0;

	b(0, 0) = 6;
	b(0, 1) = 3;
	b(0, 2) = 5;
	b(1, 0) = 3;
	b(1, 1) = 9;
	b(1, 2) = 7;
	b(2, 0) = 3;
	b(2, 1) = 8;
	b(2, 2) = 1;

	Matrix c = a * b;

	EXPECT_EQ(c(0, 0), 54);
	EXPECT_EQ(c(0, 1), 95);
	EXPECT_EQ(c(0, 2), 48);
	EXPECT_EQ(c(1, 0), 69);
	EXPECT_EQ(c(1, 1), 69);
	EXPECT_EQ(c(1, 2), 62);
	EXPECT_EQ(c(2, 0), 15);
	EXPECT_EQ(c(2, 1), 15);
	EXPECT_EQ(c(2, 2), 17);
}

TEST(MatrixOperatorTestSuit, MatrixOperatorTestSuit_Multiplication2_Test) {
	Matrix a = Matrix(2, 3);
	Matrix b = Matrix(3, 4);

	a(0, 0) = -1;
	a(0, 1) = 0;
	a(0, 2) = -3;
	a(1, 0) = 2;
	a(1, 1) = 4;
	a(1, 2) = -2;

	b(0, 0) = 3;
	b(0, 1) = 6;
	b(0, 2) = -3;
	b(0, 3) = -1;
	b(1, 0) = 0;
	b(1, 1) = 1;
	b(1, 2) = 4;
	b(1, 3) = 2;
	b(2, 0) = 7;
	b(2, 1) = -4;
	b(2, 2) = 5;
	b(2, 3) = -2;

	Matrix c = a * b;

	EXPECT_EQ(c(0, 0), -24);
	EXPECT_EQ(c(0, 1), 6);
	EXPECT_EQ(c(0, 2), -12);
	EXPECT_EQ(c(0, 3), 7);
	EXPECT_EQ(c(1, 0), -8);
	EXPECT_EQ(c(1, 1), 24);
	EXPECT_EQ(c(1, 2), 0);
	EXPECT_EQ(c(1, 3), 10);
}

TEST(MatrixOperatorTestSuit, MatrixOperatorTestSuit_Transpose_Test) {
	Matrix a = Matrix(2, 3);

	a(0, 0) = 4;
	a(0, 1) = 3;
	a(0, 2) = 7;
	a(1, 0) = 9;
	a(1, 1) = 2;
	a(1, 2) = 3;

	Matrix b = a.T();

	EXPECT_EQ(b.getRows(), a.getCols());
	EXPECT_EQ(b.getCols(), a.getRows());

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 3; ++j) {
			EXPECT_EQ(a(i, j), b(j, i));
		}
	}
}

TEST(MatrixOperatorTestSuit, MatrixOperatorTestSuit_LUDecomposition_Test) {
	Matrix a = Matrix(3, 3);

	a(0, 0) = 2;
	a(0, 1) = -1;
	a(0, 2) = -2;
	a(1, 0) = -4;
	a(1, 1) = 6;
	a(1, 2) = 3;
	a(2, 0) = -4;
	a(2, 1) = -2;
	a(2, 2) = 8;

	std::pair<Matrix, Matrix> LU = a.LUDecomposition();

	EXPECT_EQ(LU.first(0, 0), 1);
	EXPECT_EQ(LU.first(0, 1), 0);
	EXPECT_EQ(LU.first(0, 2), 0);
	EXPECT_EQ(LU.first(1, 0), -2);
	EXPECT_EQ(LU.first(1, 1), 1);
	EXPECT_EQ(LU.first(1, 2), 0);
	EXPECT_EQ(LU.first(2, 0), -2);
	EXPECT_EQ(LU.first(2, 1), -1);
	EXPECT_EQ(LU.first(2, 2), 1);

	EXPECT_EQ(LU.second(0, 0), 2);
	EXPECT_EQ(LU.second(0, 1), -1);
	EXPECT_EQ(LU.second(0, 2), -2);
	EXPECT_EQ(LU.second(1, 0), 0);
	EXPECT_EQ(LU.second(1, 1), 4);
	EXPECT_EQ(LU.second(1, 2), -1);
	EXPECT_EQ(LU.second(2, 0), 0);
	EXPECT_EQ(LU.second(2, 1), 0);
	EXPECT_EQ(LU.second(2, 2), 3);

	Matrix b = LU.first * LU.second;
	EXPECT_EQ(a, b);
}

TEST(MatrixOperatorTestSuit, MatrixOperatorTestSuit_inverse_Test) {
	Matrix a = Matrix(3, 3);

	a(0, 0) = 2;
	a(0, 1) = -1;
	a(0, 2) = -2;
	a(1, 0) = -4;
	a(1, 1) = 6;
	a(1, 2) = 3;
	a(2, 0) = -4;
	a(2, 1) = -2;
	a(2, 2) = 8;

	Matrix b = a.inverse();

	Matrix c = a * b;
	std::cout << c << std::endl;
	EXPECT_TRUE(c == IdentityMatrix(3));
}