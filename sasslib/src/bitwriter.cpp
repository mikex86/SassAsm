#include "bitwriter.h"

#include <cstdint>
#include <cstddef>

#define bitSize(x)          (sizeof(x) * 8)


BitWriter::BitWriter(void* data) : data(static_cast<uint64_t*>(data))
{
}

void BitWriter::writeBits(const uint64_t value, const uint32_t offset, const uint32_t length) const
{
    write_bits(value, offset, length, data);
}

void write_bits(const uint64_t value, const uint32_t offset, const uint32_t length, uint64_t* dst)
{
    constexpr size_t dst_bit_size = sizeof(*dst) * 8;
    const uint64_t value_mask = length == 64 ? UINT64_MAX : ~(UINT64_MAX << length);
    const uint32_t word_idx = offset / dst_bit_size;
    const uint32_t word_bit_idx = offset % dst_bit_size;
    const uint32_t bit_shift = word_bit_idx;
    const uint64_t masked_value = value & value_mask;
    if (bit_shift + length <= dst_bit_size)
    {
        const uint64_t valueHWord = masked_value << bit_shift;
        const uint64_t maskHWord = value_mask << bit_shift;
        dst[word_idx] = dst[word_idx] & ~maskHWord | valueHWord & maskHWord;
    }
    else
    {
        // field spans two words
        const uint64_t value_hi_word = masked_value << bit_shift;
        const uint64_t value_lo_word = masked_value >> dst_bit_size - bit_shift;

        const uint64_t mask_hi_word = value_mask << bit_shift;
        const uint64_t mask_lo_word = value_mask >> dst_bit_size - bit_shift;

        dst[word_idx] = dst[word_idx] & ~mask_hi_word | value_hi_word & mask_hi_word;
        dst[word_idx + 1] = dst[word_idx + 1] & ~mask_lo_word | value_lo_word & mask_lo_word;
    }
}
