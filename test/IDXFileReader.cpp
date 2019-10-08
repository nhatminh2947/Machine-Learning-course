//
// Created by nhatminh2947 on 10/8/19.
//

#include "gtest/gtest.h"
#include "IdxFileReader.h"

TEST(IDXFileReaderTestSuite, ReadFile) {
    IdxFileReader::ReadImages("../src/HW2/train-images-idx3-ubyte");
}