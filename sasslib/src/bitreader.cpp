#include "bitreader.h"

#include <cstddef>
#include <cstdint>


uint64_t read_bits(const uint32_t offset, const uint32_t length, const uint64_t* src)
{
    constexpr size_t src_bit_size = sizeof(*src) * 8;
    const uint64_t value_mask = length == 64 ? UINT64_MAX : ~(UINT64_MAX << length);
    const uint32_t code_word_offset = offset / src_bit_size;
    const uint32_t bit_in_word_offset = offset % src_bit_size;
    const uint32_t bit_shift = bit_in_word_offset;

    if (bit_shift + length <= src_bit_size)
    {
        const uint64_t maskHWord = value_mask << bit_shift;
        const uint64_t valueHWord = src[code_word_offset] & maskHWord;
        return valueHWord >> bit_shift;
    }

    // field spans two words
    const uint64_t mask_hi_word = value_mask >> (src_bit_size - bit_shift);
    const uint64_t mask_lo_word = value_mask << bit_shift;

    const uint64_t value_lo_word = src[code_word_offset] & mask_lo_word;
    const uint64_t value_hi_word = src[code_word_offset + 1] & mask_hi_word;
    return (value_hi_word << (src_bit_size - bit_shift)) | (value_lo_word >> bit_shift);
}
