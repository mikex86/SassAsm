#pragma once

#include <cstdint>
#include <sasslib.h>

#ifdef __GNUC__
#define PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
#endif

#ifdef _MSC_VER
#define PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
#endif

PACK(struct SassInstructionDataSm89 : SassInstructionData {
    uint64_t data[2];
    });

#define COMMON_SCHED_INFO() uint8_t reuse{}; uint8_t b_mask{}; uint8_t w_bar = -1; uint8_t r_bar = -1; uint8_t y{}; uint8_t stall{};

static_assert(sizeof(SassInstructionDataSm89) == 16, "Invalid size for SerializedInstruction");

// @(!)PN MOV R_IMM, C[I_IMM][J_IMM], MASK_IMM
// Move "Register", "Bank"
struct MovRegBankSm89 final : SassInstruction
{
    uint8_t dst_reg; // register
    uint8_t src_bank; // C[i]
    uint16_t src_addr; // C[...][j]

    uint16_t p = -1; // predicate
    bool p_negate = false; // predicate negate (@!PN distilled)
    uint8_t mask = 0xF;

    COMMON_SCHED_INFO();

    void serialize(SassInstructionData& dst_buf) override;

    ~MovRegBankSm89() override;
};

// @(!)PN IMAD.(MOV.)U32 R_DST, R_A, R_B, C[I_IMM][J_IMM]
// If R_A or R_B are REG_RZ, then mnemonic becomes IMAD.MOV.U32, else its IMAD.U32
struct ImadU32Sm89 final : SassInstruction
{
    uint8_t dst; // R_DST
    uint8_t src_a; // R_A
    uint8_t src_b; // R_B

    uint8_t src_bank; // C[I_IMM]
    uint16_t src_addr; // C[...][J_IMM]

    uint16_t p = -1; // predicate
    bool p_negate = false; // predicate negate (@!PN distilled)

    COMMON_SCHED_INFO();

    void serialize(SassInstructionData& dst_buf) override;

    ~ImadU32Sm89() override;
};

enum class DtypeSm89
{
    U8 = 0, S8 = 1, U16 = 2, S16 = 3, U32 = 4, U64 = 5
};

// @(!)PN ULDC.{U8, S8, U16, S16, 32, 64} UR_DST, C[I_IMM][J_IMM]
struct UldcRegBankSm89 final : SassInstruction
{
    uint8_t dst; // UR_DST
    DtypeSm89 dtype; // .U8, S8... ; U32 is default and implicit in mnemonic

    uint8_t src_bank; // C[I_IMM]
    uint16_t src_addr; // C[...][J_IMM]

    uint16_t up = -1; // predicate
    bool up_negate = false; // predicate negate (@!UPN distilled)

    COMMON_SCHED_INFO();

    void serialize(SassInstructionData& dst_buf) override;

    ~UldcRegBankSm89() override;
};

enum class ModeSm89 // TODO: Don't have a better name for this right now...
{
    EF = 0, NONE = 1, EL = 2, LU = 3, EU = 4, NA = 5
};

// @(!)PN LDG.(E.)({EL, EF, ...})(.{U8, ...} if dtype != U32) (P_DST if P_DST != 7), R_DST, [ R_SRC.{32, 64} + (R_OFF if no_regoffcalc == false) + (IMM_OFF if IMM_OFF != 0) ]
struct LdgUrSm89 final : SassInstruction
{
    uint8_t pred_dst = -1; // P_DST ; -1 == 7 due to bit trunc.
    uint8_t src; // R_SRC
    uint8_t src_offreg; // R_OFF
    uint8_t src_offimm; // IMM_OFF
    uint8_t dst; // R_DST
    bool src_is64bit;

    bool no_regoffcalc; // disables the addition of R_OFF

    bool is_e; // controls presence of .E
    DtypeSm89 dtype = DtypeSm89::U32; // .U8, S8... ; U32 is default and implicit in mnemonic

    ModeSm89 mode = ModeSm89::NONE; // controls EF, EL, LU, NA or non presence if NONE

    uint16_t p = -1; // predicate
    bool p_negate = false; // predicate negate (@!PN distilled)

    COMMON_SCHED_INFO();

    void serialize(SassInstructionData& dst_buf) override;

    ~LdgUrSm89() override;
};

// @(!)PN STG.(E.)({EL, EF, ...})(.{U8, ...} if dtype != U32) [ R_DST.64 + (R_OFF if no_regoffcalc == false) + (IMM_OFF if IMM_OFF != 0) ], R_SRC
struct StgUrSm89 final : SassInstruction
{
    uint8_t dst; // R_DST
    uint8_t src; // R_SRC
    uint8_t dst_offimm; // IMM_OFF
    uint8_t dst_offreg; // R_OFF

    bool dst_is64bit = true; // false is illegal here because dst must be an address...

    bool no_regoffcalc; // disables the addition of R_OFF

    bool is_e; // controls presence of .E
    DtypeSm89 dtype = DtypeSm89::U32; // .U8, S8... ; U32 is default and implicit in mnemonic

    ModeSm89 mode = ModeSm89::NONE; // controls EF, EL, LU, NA or non presence if NONE

    uint16_t p = -1; // predicate
    bool p_negate = false; // predicate negate (@!PN distilled)

    COMMON_SCHED_INFO();

    void serialize(SassInstructionData& dst_buf) override;

    ~StgUrSm89() override;
};

// @(!)PN S2R R_DST, SRC_SR
struct S2rSm89 final : SassInstruction
{
    uint8_t dst; // R_DST
    uint8_t src; // SRC_SR

    uint16_t p = -1; // predicate
    bool p_negate = false; // predicate negate (@!PN distilled)

    COMMON_SCHED_INFO();

    void serialize(SassInstructionData& dst_buf) override;

    ~S2rSm89() override;
};


// @(!)PN EXIT
struct ExitSm89 final : SassInstruction
{
    uint16_t p = -1; // predicate
    bool p_negate = false; // predicate negate (@!PN distilled)

    COMMON_SCHED_INFO();

    void serialize(SassInstructionData& dst_buf) override;

    ~ExitSm89() override;
};

// @(!)PN BRA IMM_OFF
struct BraSm89 final : SassInstruction
{
    int64_t off_imm;

    uint16_t p = -1; // predicate
    bool p_negate = false; // predicate negate (@!PN distilled)

    COMMON_SCHED_INFO();

    void serialize(SassInstructionData& dst_buf) override;

    ~BraSm89() override;
};

// NOP
struct NopSm89 final : SassInstruction
{
    uint16_t p = -1; // predicate
    bool p_negate = false; // predicate negate (@!PN distilled)

    COMMON_SCHED_INFO();

    void serialize(SassInstructionData& dst_buf) override;

    ~NopSm89() override;
};
