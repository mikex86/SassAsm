#ifndef SASS_H
#define SASS_H

#include <cstdio>

typedef __uint128_t sass_inst_buf;

inline void debug_print_hex_bytes(const sass_inst_buf &buf) {
    for (int i = 0; i < 16; i++) {
        const uint8_t byte = (buf >> (128 - 8 * (i + 1))) & 0xFF;
        printf("%02x ", byte);
    }
    printf("\n");
}

class SassInstWriter {
    sass_inst_buf &inst;
    size_t bit_index;

public:
    explicit SassInstWriter(sass_inst_buf &inst) : inst(inst), bit_index(0) {
    }
};

#endif //SASS_H
