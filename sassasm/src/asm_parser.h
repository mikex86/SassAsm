#ifndef ASM_PARSER_H
#define ASM_PARSER_H

#include <iostream>
#include <optional>
#include <string>
#include <cstdint>

#include "special_registers.h"

/**
 * Asserts a condition and aborts the assembler with a detailed error message if the condition is false.
 */
void compiler_assert(bool condition,
                     const std::string& message,
                     const std::string& file_name, const std::string& line,
                     int line_nr, int col_nr);

#define COMPILER_ASSERT(condition, message, file_name, line, line_nr, col_nr) { bool __assert_cond = condition; compiler_assert(__assert_cond, message, file_name, line, line_nr, col_nr); assert(__assert_cond); }

/**
 * Expect an identifier at the current position in the line.
 * If not a single identifier character is found, an empty string is returned.
 * This must be checked by the caller.
 */
std::string expect_identifier(const std::string& line, int& col_nr);

std::string expect_string_literal(const std::string& line, int& col_nr);

std::string expect_string_literal_or_identifier(const std::string& line, int& col_nr);

std::string expect_directive_or_label_or_inst(const std::string& file_name, const std::string& line,
                                              int line_nr, int& col_nr);

uint32_t expect_register_identifier(const std::string& file_name, const std::string& line, int line_nr,
                                    int& col_nr);

uint32_t expect_predicate_identifier(const std::string& file_name, const std::string& line, int line_nr,
                                     int& col_nr);

uint32_t expect_address_literal(const std::string& file_name, const std::string& line, int line_nr,
                                int& col_nr);

void expect_parameter_delimiter(const std::string& file_name, const std::string& line, int line_nr,
                                int& col_nr);

struct ConstantLoadExpr
{
    uint8_t memory_bank; // Max: 31
    uint64_t memory_address;
};

ConstantLoadExpr expect_constant_load_expr(const std::string& file_name, const std::string& line, int line_nr,
                                           int& col_nr);

/**
 * A fused register offset expression is an expression of the form [REG_BASE + REG_OFFSET + IMM_OFFSET], which expresses
 * the computed result of this expression as a single operand to an instruction.
 */
struct FusedRegisterOffsetExpr
{
    uint32_t base_register = -1;
    uint32_t offset_register = -1;
    uint64_t offset_imm = 0;
    bool base_reg_64_bit = false;
};

FusedRegisterOffsetExpr expect_fused_register_offset_expr(const std::string& file_name, const std::string& line,
                                                          int line_nr, int& col_nr);

std::optional<uint64_t> expect_uint_literal(const std::string& line, int& col_nr);

std::optional<SpecialRegisterExpression> expect_special_register(const std::string& line, int& col_nr);

void expect_space(const std::string& file_name, const std::string& line, int line_nr, int& col_nr);

void expect_statement_end(const std::string& file_name, const std::string& line, int line_nr, int& col_nr);

#endif //ASM_PARSER_H
