project(sasslib CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(sasslib src/sasslib.cpp src/sm_89.cpp src/bitwriter.cpp
        src/bitreader.cpp
        internal/bitreader.h
)

target_include_directories(sasslib PUBLIC include)
target_include_directories(sasslib PRIVATE internal)