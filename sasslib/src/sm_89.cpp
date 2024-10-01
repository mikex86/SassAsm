#include "sm_89.h"

#include <bitwriter.h>

uint32_t encode_sched_info(const uint8_t reuse, const uint8_t b_mask, const uint8_t w_bar, const uint8_t r_bar,
                           const uint8_t y, const uint8_t stall)
{
    uint32_t scheduling_info = 0;
    scheduling_info |= (reuse & 0xF) << 17;
    scheduling_info |= (b_mask & 0x3F) << 11;
    scheduling_info |= (w_bar & 0x7) << 8;
    scheduling_info |= (r_bar & 0x7) << 5;
    scheduling_info |= (y & 0x1) << 4;
    scheduling_info |= (stall & 0xF);
    return scheduling_info;
}


// MOV
void MovRegBankSm89::serialize(SassInstructionData& dst_buf)
{
    const BitWriter writer(&dst_buf);

    writer.writeBits(dst_reg, 16, 8);

    writer.writeBits(src_bank, 54, 5);

    // even though the address is 64 bits, >> 2 makes it express up to 65536, which seems to be the max for constant banks
    writer.writeBits(src_addr >> 2, 40, 14);

    writer.writeBits(mask, 72, 4);

    // common
    writer.writeBits(p, 12, 3);
    writer.writeBits(p_negate, 15, 1);
    writer.writeBits(encode_sched_info(reuse, b_mask, w_bar, r_bar, y, stall), 105, 21);

    // random fuzzed constants
    writer.writeBits(0x2, 0, 5);
    writer.writeBits(0x0, 5, 3);
    writer.writeBits(0x0, 8, 1);
    writer.writeBits(0x5, 9, 3);
}

MovRegBankSm89::~MovRegBankSm89() = default;


// IMAD
void ImadU32Sm89::serialize(SassInstructionData& dst_buf)
{
    const BitWriter writer(&dst_buf);

    writer.writeBits(src_bank, 54, 5);

    // even though the address is 64 bits, >> 2 makes it express up to 65536, which seems to be the max for constant banks
    writer.writeBits(src_addr >> 2, 40, 14);

    writer.writeBits(dst, 16, 8);
    writer.writeBits(src_a, 24, 8);
    writer.writeBits(src_b, 64, 8);

    // common
    writer.writeBits(p, 12, 3);
    writer.writeBits(p_negate, 15, 1);
    writer.writeBits(encode_sched_info(reuse, b_mask, w_bar, r_bar, y, stall), 105, 21);

    // random fuzzed
    writer.writeBits(0x4, 0, 5);
    writer.writeBits(0x1, 5, 3);
    writer.writeBits(0x0, 8, 1);
    writer.writeBits(0x3, 9, 3);
}

ImadU32Sm89::~ImadU32Sm89() = default;

void UldcRegBankSm89::serialize(SassInstructionData& dst_buf)
{
    const BitWriter writer(&dst_buf);

    writer.writeBits(static_cast<uint64_t>(dtype), 73, 3);

    writer.writeBits(dst, 16, 6);

    writer.writeBits(src_bank, 54, 5);

    // even though the address is 64 bits, >> 2 makes it express up to 65536, which seems to be the max for constant banks
    writer.writeBits(src_addr >> 2, 40, 14);

    // common
    writer.writeBits(up, 12, 3);
    writer.writeBits(up_negate, 15, 1);
    writer.writeBits(encode_sched_info(reuse, b_mask, w_bar, r_bar, y, stall), 105, 21);

    // random fuzzed constants
    writer.writeBits(0x19, 0, 5);
    writer.writeBits(0x5, 5, 3);
    writer.writeBits(0x0, 8, 1);
    writer.writeBits(0x5, 9, 3);
}

UldcRegBankSm89::~UldcRegBankSm89() = default;

void LdgUrSm89::serialize(SassInstructionData& dst_buf)
{
    const BitWriter writer(&dst_buf);

    writer.writeBits(pred_dst, 81, 3);
    writer.writeBits(src, 24, 8);
    writer.writeBits(src_offreg, 32, 8);
    writer.writeBits(src_offimm, 40, 24);
    writer.writeBits(dst, 16, 8);

    writer.writeBits(src_is64bit, 90, 1);

    // common
    writer.writeBits(is_e, 72, 1);
    writer.writeBits(static_cast<uint64_t>(dtype), 73, 3);
    writer.writeBits(static_cast<uint64_t>(mode), 84, 3);

    writer.writeBits(p, 12, 3);
    writer.writeBits(p_negate, 15, 1);
    writer.writeBits(encode_sched_info(reuse, b_mask, w_bar, r_bar, y, stall), 105, 21);

    writer.writeBits(no_regoffcalc, 76, 1);

    // random fuzzed constants
    writer.writeBits(0x1, 0, 5);
    writer.writeBits(0x4, 5, 3);
    writer.writeBits(0x1, 8, 1);
    writer.writeBits(0x4, 9, 3);
    writer.writeBits(0x1, 91, 1); // is_ur=true
}

LdgUrSm89::~LdgUrSm89() = default;

void StgUrSm89::serialize(SassInstructionData& dst_buf)
{
    const BitWriter writer(&dst_buf);

    writer.writeBits(dst, 24, 8);
    writer.writeBits(src, 32, 8);
    writer.writeBits(dst_offreg, 64, 8);
    writer.writeBits(dst_offimm, 40, 24);
    writer.writeBits(dst_is64bit, 90, 1);

    // common
    writer.writeBits(is_e, 72, 1);
    writer.writeBits(static_cast<uint64_t>(dtype), 73, 3);
    writer.writeBits(static_cast<uint64_t>(mode), 84, 3);

    writer.writeBits(p, 12, 3);
    writer.writeBits(p_negate, 15, 1);
    writer.writeBits(encode_sched_info(reuse, b_mask, w_bar, r_bar, y, stall), 105, 21);

    writer.writeBits(no_regoffcalc, 76, 1);

    // random fuzzed constants
    writer.writeBits(0x6, 0, 5);
    writer.writeBits(0x4, 5, 3);
    writer.writeBits(0x1, 8, 1);
    writer.writeBits(0x4, 9, 3);
    writer.writeBits(0x1, 91, 1); // is_ur=true
}

StgUrSm89::~StgUrSm89() = default;

void S2rSm89::serialize(SassInstructionData& dst_buf)
{
    const BitWriter writer(&dst_buf);

    // common
    writer.writeBits(p, 12, 3);
    writer.writeBits(p_negate, 15, 1);
    writer.writeBits(encode_sched_info(reuse, b_mask, w_bar, r_bar, y, stall), 105, 21);

    writer.writeBits(dst, 16, 8);

    writer.writeBits(src, 72, 8);

    // random fuzzed constants
    writer.writeBits(0x4, 9, 3);
    writer.writeBits(0x1, 8, 1);
    writer.writeBits(0x0, 5, 3);
    writer.writeBits(0x19, 0, 5);
}

S2rSm89::~S2rSm89() = default;


void ExitSm89::serialize(SassInstructionData& dst_buf)
{
    const BitWriter writer(&dst_buf);

    writer.writeBits(p, 12, 3);
    writer.writeBits(p_negate, 15, 1);
    writer.writeBits(encode_sched_info(reuse, b_mask, w_bar, r_bar, y, stall), 105, 21);

    // TODO: figure out what this is
    //  This looks like some mask of a currently unknown field
    //  that exit does not make meaningful use of
    writer.writeBits(7, 87, 3);

    // random fuzzed constants
    writer.writeBits(0x4, 9, 3);
    writer.writeBits(0x1, 8, 1);
    writer.writeBits(0x2, 5, 3);
    writer.writeBits(0xd, 0, 5);
}

ExitSm89::~ExitSm89() = default;

void BraSm89::serialize(SassInstructionData& dst_buf)
{
    const BitWriter writer(&dst_buf);

    writer.writeBits((1ull << 48) + (off_imm >> 2) & ((1ull << 48) - 1), 34, 48);

    // TODO: figure out what this is
    //  This looks like some mask of a currently unknown field
    //  that exit does not make meaningful use of
    writer.writeBits(7, 87, 3);

    // common
    writer.writeBits(p, 12, 3);
    writer.writeBits(p_negate, 15, 1);
    writer.writeBits(encode_sched_info(reuse, b_mask, w_bar, r_bar, y, stall), 105, 21);

    // random fuzzed constants
    writer.writeBits(0x7, 0, 5);
    writer.writeBits(0x2, 5, 3);
    writer.writeBits(0x1, 8, 1);
    writer.writeBits(0x4, 9, 3);
}

BraSm89::~BraSm89() = default;


void NopSm89::serialize(SassInstructionData& dst_buf)
{
    const BitWriter writer(&dst_buf);

    // common
    writer.writeBits(p, 12, 3);
    writer.writeBits(p_negate, 15, 1);
    writer.writeBits(encode_sched_info(reuse, b_mask, w_bar, r_bar, y, stall), 105, 21);

    writer.writeBits(0x4, 9, 3);
    writer.writeBits(0x1, 8, 1);
    writer.writeBits(0x0, 5, 3);
    writer.writeBits(0x18, 0, 5);
}

NopSm89::~NopSm89() = default;
