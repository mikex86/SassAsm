project(sassasm CXX)

set(CMAKE_CXX_STANDARD 20)

add_executable(sassasm src/main.cpp
        src/asm_parser.cpp
        src/asm_parser.h
        src/cuelf.h
        src/patch_elf.cpp
        src/patch_elf.h
        src/tinyscript.cpp
        src/tinyscript.h)

target_link_libraries(sassasm PRIVATE sasslib elfio::elfio)