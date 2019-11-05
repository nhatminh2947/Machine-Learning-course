project(Homework)

set(HEADER_FILES
        matrix.h
        Point.h
        IdxFileReader.h
        Col.h
        )

set(SOURCE_FILES
        matrix.cpp
        Point.cpp
        IdxFileReader.cpp
        Col.cpp)

add_library(MatrixLib STATIC ${SOURCE_FILES} ${HEADER_FILES})

set(RANDOM_HEADER_FILES
        GaussianDataGenerator.h
        PolynomialDataGenerator.h
        )

set(RANDOM_SOURCE_FILES
        GaussianDataGenerator.cpp
        PolynomialDataGenerator.cpp)

add_library(RandomLib STATIC ${RANDOM_SOURCE_FILES} ${RANDOM_HEADER_FILES})

set(METRIC_HEADER_FILES
        ConfusionMatrix.h
        )

set(METRIC_SOURCE_FILES
        ConfusionMatrix.cpp)

add_library(MetricLib STATIC ${METRIC_SOURCE_FILES} ${METRIC_HEADER_FILES})