//
// Created by nhatminh2947 on 10/8/19.
//

#ifndef MACHINE_LEARNING_COURSE_IDXFILEREADER_H
#define MACHINE_LEARNING_COURSE_IDXFILEREADER_H


#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "matrix.h"

class IdxFileReader {
private:
	static int reverse(int i);

public:
    static Matrix<double> ReadImages(const std::string &path, int n_data);

    static std::vector<int> ReadLabels(const std::string &path, int n_data);
};


#endif //MACHINE_LEARNING_COURSE_IDXFILEREADER_H

