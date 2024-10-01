#include "asm_parser.h"

#include <cassert>
#include <sasslib.h>

void compiler_assert(const bool condition, const std::string& message, const std::string& file_name,
                     const std::string& line, const int line_nr, const int col_nr)
{
    if (!condition)
    {
        // print error message
        std::cout << file_name << ":" << line_nr << "\033[1;31m error: \033[0m" << message << std::endl;
        std::cout << line << std::endl;

        // print error position
        for (int i = 0; i < col_nr; i++)
        {
            std::cout << " ";
        }
        std::cout << "\033[1;31m^\033[0m" << std::endl;
        exit(1);
    }
}


std::string expect_identifier(const std::string& line, int& col_nr)
{
    // iterate over the line until we find a non-identifier character
    std::string identifier;
    while (col_nr < line.size() && (line[col_nr] >= 'a' && line[col_nr] <= 'z') ||
        (line[col_nr] >= 'A' && line[col_nr] <= 'Z') ||
        (line[col_nr] >= '0' && line[col_nr] <= '9') ||
        line[col_nr] == '_' || line[col_nr] == '.')
    {
        identifier += line[col_nr];
        col_nr++;
    }
    return identifier;
}

std::string expect_string_literal(const std::string& line, int& col_nr)
{
    // expect a double quote
    COMPILER_ASSERT(line[col_nr] == '"', "expected double quote", "", line, 0, col_nr);
    col_nr++;

    // iterate over the line until we find the closing double quote
    std::string literal;
    while (col_nr < line.size() && line[col_nr] != '"')
    {
        literal += line[col_nr];
        col_nr++;
    }

    // expect a closing double quote
    COMPILER_ASSERT(col_nr < line.size() && line[col_nr] == '"', "expected closing double quote", "", line, 0, col_nr);
    col_nr++;

    return literal;
}

std::string expect_string_literal_or_identifier(const std::string& line, int& col_nr)
{
    // if next character is a double quote, we have a string literal
    if (line[col_nr] == '"')
    {
        return expect_string_literal(line, col_nr);
    }
    // else we have an identifier
    return expect_identifier(line, col_nr);
}

std::string expect_directive_or_label_or_inst(const std::string& file_name, const std::string& line, const int line_nr,
                                              int& col_nr)
{
    // expect either
    // - IDENTIFIER -> directive
    // - IDENTIFIER, COLON -> label
    std::string identifier = expect_identifier(line, col_nr);

    COMPILER_ASSERT(!identifier.empty(), "expected identifier", file_name, line, line_nr, col_nr);

    // if next character is a colon, we have a label
    if (line[col_nr] == ':')
    {
        col_nr++;
        return identifier + ':';
    }

    // if first character of identifier is a dot, we have a directive
    if (identifier[0] == '.')
    {
        COMPILER_ASSERT(line[col_nr] == ' ', "expected space after directive", file_name, line, line_nr, col_nr);
        return identifier;
    }

    // else we have an instruction
    COMPILER_ASSERT(col_nr == line.size() || line[col_nr] == ' ' || line[col_nr] == ';', "expected space/semi colon after instruction mnemonic",
                    file_name, line, line_nr, col_nr);
    return identifier;
}

uint32_t expect_register_identifier(const std::string& file_name, const std::string& line, const int line_nr,
                                    int& col_nr)
{
    // expect RN where N is a number
    COMPILER_ASSERT(line[col_nr] == 'R', "expected register identifier of the form RN", file_name, line, line_nr,
                    col_nr);
    col_nr++;

    if (line[col_nr] == 'Z')
    {
        col_nr++;
        return REG_RZ;
    }

    const auto reg_literal = expect_uint_literal(line, col_nr);
    COMPILER_ASSERT(reg_literal.has_value(), "expected register number", file_name, line, line_nr, col_nr);
    return reg_literal.value();
}

uint32_t expect_predicate_identifier(const std::string& file_name, const std::string& line, const int line_nr,
                                     int& col_nr)
{
    // expect RN where N is a number
    COMPILER_ASSERT(line[col_nr] == 'P', "expected predicate identifier of the form PN", file_name, line, line_nr,
                    col_nr);
    col_nr++;

    const auto reg_literal = expect_uint_literal(line, col_nr);
    COMPILER_ASSERT(reg_literal.has_value(), "expected predicate numeral", file_name, line, line_nr, col_nr);
    return reg_literal.value();
}

uint32_t expect_address_literal(const std::string& file_name, const std::string& line, const int line_nr, int& col_nr)
{
    const auto addr_literal = expect_uint_literal(line, col_nr);
    COMPILER_ASSERT(addr_literal.has_value(), "expected predicate numeral", file_name, line, line_nr, col_nr);
    return addr_literal.value();
}

void expect_parameter_delimiter(const std::string& file_name, const std::string& line, const int line_nr,
                                int& col_nr)
{
    // expect an arbitrary number of white characters
    while (col_nr < line.size() && std::isspace(line[col_nr]))
    {
        col_nr++;
    }

    // expect one comma
    COMPILER_ASSERT(line[col_nr] == ',', "expected comma", file_name, line, line_nr, col_nr);
    col_nr++;

    // expect an arbitrary number of white characters
    while (col_nr < line.size() && std::isspace(line[col_nr]))
    {
        col_nr++;
    }
}


ConstantLoadExpr expect_constant_load_expr(const std::string& file_name, const std::string& line, const int line_nr,
                                           int& col_nr)
{
    // Expect C[bank][address]
    COMPILER_ASSERT(line[col_nr] == 'C' || line[col_nr] == 'c',
                    "expected constant load expression of the form C[bank][address]", file_name,
                    line, line_nr, col_nr);
    col_nr++;

    // Expect '['
    COMPILER_ASSERT(line[col_nr] == '[', "expected '[' in constant load expression", file_name, line, line_nr, col_nr);
    col_nr++;

    // Expect bank number
    const uint32_t bank = expect_uint_literal(line, col_nr).value();
    COMPILER_ASSERT(bank <= 31, "bank number must be in the range [0, 31]", file_name, line, line_nr, col_nr);

    // Expect ']'
    COMPILER_ASSERT(line[col_nr] == ']', "expected ']' in constant load expression", file_name, line, line_nr, col_nr);
    col_nr++;

    // Expect '['
    COMPILER_ASSERT(line[col_nr] == '[', "expected '[' in constant load expression", file_name, line, line_nr, col_nr);
    col_nr++;

    // Expect address
    const uint32_t address = expect_uint_literal(line, col_nr).value();

    // Expect ']'
    COMPILER_ASSERT(line[col_nr] == ']', "expected ']' in constant load expression", file_name, line, line_nr, col_nr);
    col_nr++;

    return ConstantLoadExpr{
        .memory_bank = static_cast<uint8_t>(bank),
        .memory_address = address
    };
}

FusedRegisterOffsetExpr expect_fused_register_offset_expr(const std::string& file_name, const std::string& line, const int line_nr, int& col_nr)
{
    // Expect '['
    COMPILER_ASSERT(line[col_nr] == '[', "expected '[' in constant load expression", file_name, line, line_nr, col_nr);
    col_nr++;

    // Expect base register
    const uint32_t base_register = expect_register_identifier(file_name, line, line_nr, col_nr);

    uint32_t offset_reg = -1;
    uint64_t offset_imm = 0;
    bool base_reg_64_bit = false;

    // check if there is a dtype qualifier
    if (col_nr < line.size() && line[col_nr] == '.')
    {
        col_nr++;
        std::string dtype_qualifier;
        while (col_nr < line.size() && line[col_nr] != '.' && line[col_nr] != ']' && line[col_nr] != '+' && line[col_nr]
            != ' ')
        {
            dtype_qualifier += line[col_nr];
            col_nr++;
        }
        if (dtype_qualifier == "64")
        {
            base_reg_64_bit = true;
        }
    }

    // expect an arbitrary number of white characters
    while (col_nr < line.size() && std::isspace(line[col_nr]))
    {
        col_nr++;
    }

    // offset register
    if (col_nr < line.size() && line[col_nr] == '+')
    {
        col_nr++;
        offset_reg = expect_register_identifier(file_name, line, line_nr, col_nr);
    }

    // offset immediate
    if (col_nr < line.size() && line[col_nr] == '+')
    {
        col_nr++;
        const auto literal = expect_uint_literal(line, col_nr);
        COMPILER_ASSERT(literal.has_value(), "expected immediate offset literal", file_name, line, line_nr, col_nr);
        offset_imm = literal.value();
    }

    // expect ']'
    COMPILER_ASSERT(line[col_nr] == ']', "expected ']' in UR expression", file_name, line, line_nr, col_nr);
    col_nr++;

    return FusedRegisterOffsetExpr{
        .base_register = base_register,
        .offset_register = offset_reg,
        .offset_imm = offset_imm,
        .base_reg_64_bit = base_reg_64_bit
    };
}

std::optional<uint64_t> expect_uint_literal(const std::string& line, int& col_nr)
{
    // check if is hex literal
    bool is_hex = false;
    if (col_nr + 2 < line.size() && line[col_nr] == '0' && (line[col_nr + 1] == 'x'))
    {
        col_nr += 2;
        is_hex = true;
    }

    std::string literal;
    while (col_nr < line.size())
    {
        const char c = line[col_nr];
        const bool is_dec_digit = c >= '0' && c <= '9';

        if (const bool is_hex_ext_digit = c >= 'a' && c <= 'f';
            is_hex && !is_hex_ext_digit && !is_dec_digit)
        {
            break;
        }
        if (!is_hex && !is_dec_digit)
        {
            break;
        }
        literal += c;
        col_nr++;
    }
    return literal.empty() ? std::nullopt : std::optional(std::stoull(literal, nullptr, is_hex ? 16 : 10));
}

std::optional<SpecialRegisterExpression> expect_special_register(const std::string& line, int& col_nr)
{
    // expect identifier
    std::string identifier = expect_identifier(line, col_nr);

    // if we are at a . character, we expect a suffix
    if (col_nr < line.size() && line[col_nr] == '.')
    {
        col_nr++;
        const std::string suffix = expect_identifier(line, col_nr);
        identifier += "." + suffix;
    }

    // check if identifier is a special register
    if (const auto it = asm_literal_to_sr.find(identifier); it != asm_literal_to_sr.end())
    {
        return it->second;
    }

    return std::nullopt;
}

void expect_space(const std::string& file_name, const std::string& line, const int line_nr, int& col_nr)
{
    // expect an arbitrary number of white characters
    bool at_least_one_space = false;
    while (col_nr < line.size() && std::isspace(line[col_nr]))
    {
        col_nr++;
        at_least_one_space = true;
    }
    COMPILER_ASSERT(at_least_one_space, "expected space", file_name, line, line_nr, col_nr);
}

void expect_statement_end(const std::string& file_name, const std::string& line, const int line_nr, int &col_nr)
{
    // expect an arbitrary number of white characters
    while (col_nr < line.size() && std::isspace(line[col_nr]))
    {
        col_nr++;
    }

    if (line[col_nr] == ';')
    {
        col_nr++;

        // expect an arbitrary number of white characters
        while (col_nr < line.size() && std::isspace(line[col_nr]))
        {
            col_nr++;
        }
    }
    COMPILER_ASSERT(col_nr == line.size(), "unexpected characters at end of line", file_name, line, line_nr, col_nr);
}
