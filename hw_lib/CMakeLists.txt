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