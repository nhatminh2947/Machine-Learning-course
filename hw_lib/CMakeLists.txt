project(Homework)

set(HEADER_FILES
        matrix.h
        Dataset.h
        )

set(SOURCE_FILES
        matrix.cpp
        Dataset.cpp)

add_library(Matrix_lib STATIC ${SOURCE_FILES} ${HEADER_FILES})