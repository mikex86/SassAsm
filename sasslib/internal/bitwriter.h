#pragma once

#include <cstdint>

class BitWriter {
    uint64_t *data;

public:
    explicit BitWriter(void *data);

    void writeBits(uint64_t value, uint32_t offset, uint32_t length) const;
};

void write_bits(uint64_t value, uint32_t offset, uint32_t length, uint64_t *dst);
