project(Homework)

set(HEADER_FILES
        matrix.h
        Dataset.h
        IdxFileReader.h
        )

set(SOURCE_FILES
        matrix.cpp
        Dataset.cpp
        IdxFileReader.cpp)

add_library(Matrix_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})

set(RANDOM_HEADER_FILES
        GaussianDataGenerator.h
        PolynomialDataGenerator.h
        )

set(RANDOM_SOURCE_FILES
        GaussianDataGenerator.cpp
        PolynomialDataGenerator.cpp)

add_library(Random_lib STATIC ${RANDOM_SOURCE_FILES} ${RANDOM_HEADER_FILES})