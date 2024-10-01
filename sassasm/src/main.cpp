#include <cassert>
#include <fstream>
#include <iostream>
#include <ranges>
#include <sm_89.h>
#include <sstream>

#include <elfio/elfio.hpp>

#include "asm_parser.h"
#include "cuelf.h"
#include "patch_elf.h"
#include "tinyscript.h"

std::string sanitize(const std::string& line)
{
    // remove leading and trailing whitespaces
    std::string sanitized = line;
    sanitized.erase(0, sanitized.find_first_not_of(" \t\n\r\f\v"));
    sanitized.erase(sanitized.find_last_not_of(" \t\n\r\f\v") + 1);

    // replace tabs with spaces
    for (char& i : sanitized)
    {
        if (i == '\t')
        {
            i = ' ';
        }
    }

    // replace repeated spaces with a single space
    for (size_t i = 0; i < sanitized.size(); i++)
    {
        if (sanitized[i] == ' ' && sanitized[i + 1] == ' ')
        {
            sanitized.erase(i, 1);
            i--;
        }
    }
    return sanitized;
}

/**
 * State related to the current symbol which might get created if create_symbol is true.
 */
struct SymbolState
{
    std::string symbol_name;
    ELFIO::Elf64_Addr value;
    ELFIO::Elf_Xword size = 0;
    unsigned char bind;
    unsigned char type;
    unsigned char other;
    ELFIO::Elf_Half shnd;

    bool create_symbol = false;
};

/**
 * State related to the current section processed by the assembler.
 */
struct CurrentSectionState
{
    SymbolState symbol_state{};
};

struct AsmLabel
{
    std::string name;

    /**
     * Offset from current section start in bytes
     */
    ELFIO::Elf64_Addr section_offset;
};

/**
 * Represents an unresolved symbol occurrence in the assembly code.
 * Points to a 32-bit integer in the data section of a specific section.
 * Effectively a uint32_t*.
 */
struct UnresolvedSymbolOccurrence
{
    uint32_t section_idx;
    uint64_t data_offset;
};

void assemble_file(const std::string& file_name, const std::string& sass_asm, const std::string& arch)
{
    ELFIO::elfio writer{};

    writer.create(ELFIO::ELFCLASS64, ELFIO::ELFDATA2LSB);

    // cuda specific attributes that are always the same
    writer.set_os_abi(0x33);
    writer.set_abi_version(0x7);
    writer.set_machine(ELFIO::EM_CUDA);

    ELFIO::section* strtab_sec = writer.sections.add(".strtab");
    strtab_sec->set_type(ELFIO::SHT_STRTAB);
    strtab_sec->set_flags(0);
    strtab_sec->set_addr_align(1);

    ELFIO::string_section_accessor str_writer(strtab_sec);

    // create symbol table
    ELFIO::section* symtab_sec = writer.sections.add(".symtab");
    symtab_sec->set_type(ELFIO::SHT_SYMTAB);
    symtab_sec->set_link(strtab_sec->get_index());
    symtab_sec->set_addr_align(4);
    symtab_sec->set_entry_size(writer.get_default_entry_size(ELFIO::SHT_SYMTAB));

    // Get the symbol table writer
    ELFIO::symbol_section_accessor symbols(writer, symtab_sec);

    // iterate over lines
    std::string line;
    std::istringstream stream(sass_asm);

    ELFIO::section* current_section = nullptr;
    CurrentSectionState current_section_state = {};

    int line_nr = 0;
    std::unordered_map<std::string, ELFIO::Elf_Word> symbol_indices{};
    std::unordered_map<std::string, std::vector<UnresolvedSymbolOccurrence>> unresolved_symbols{};

    std::unordered_map<std::string, ELFIO::section*> text_sections{};
    std::unordered_map<std::string, ELFIO::section*> constant_bank_sections{};
    std::unordered_map<std::string, ELFIO::section*> info_sections{};

    std::vector<AsmLabel> labels;

    // read labels in first pass
    uint64_t curr_addr = 0;
    while (true)
    {
        bool is_eof = !std::getline(stream, line);
        line_nr++;

        if (is_eof)
        {
            break;
        }

        {
            if (line.empty())
            {
                continue;
            }

            for (size_t i = 0; i < line.size() - 1; i++)
            {
                // strip everything after comment identifiers
                if (line[i] == '/' && line[i + 1] == '/' || line[i] == ';')
                {
                    line = line.substr(0, i);
                    break;
                }

                // strip everything between /* and */
                if (line[i] == '/' && line[i + 1] == '*')
                {
                    if (size_t end_comment = line.find("*/", i + 2); end_comment != std::string::npos)
                    {
                        line = line.substr(0, i) + line.substr(end_comment + 2);
                        i--;
                    }
                    else
                    {
                        COMPILER_ASSERT(false, "Unclosed comment", file_name, line, line_nr, i);
                    }
                }
            }

            // ignore empty lines
            if (line.empty())
            {
                continue;
            }

            line = sanitize(line);

            if (line.empty())
            {
                continue;
            }

            for (size_t i = 0; i < line.size() - 1; i++)
            {
                if ((line[i] == '`' && line[i + 1] == '(') ||
                    // paren is only evaluated for labels on .byte, .short, .word directives
                    ((line.find(".byte") == 0 || line.find(".short") == 0 || line.find(".word") == 0 || line.
                            find(".size") == 0) && line[i] != '@'
                        && line[i + 1] == '('))
                {
                    // find closing parenthesis
                    size_t end_expr = line.find(')', i + 2);
                    COMPILER_ASSERT(end_expr != std::string::npos, "Unclosed label expression", file_name, line,
                                    line_nr,
                                    i);
                    line = line.substr(0, i + (line[i] == '`' ? 0 : 1)) + "1" + line.substr(end_expr + 1);
                    // dummy value
                }
            }
        }

        int col_nr = 0;

        // check if next character is @, if so, handle p
        if (line[col_nr] == '@')
        {
            col_nr++;

            // optionally expect ! for p_negate
            if (line[col_nr] == '!')
            {
                col_nr++;
            }

            // Expect P
            COMPILER_ASSERT(line[col_nr] == 'P', "expected P after @ in instruction", file_name, line, line_nr, col_nr);
            col_nr++;

            auto p = expect_uint_literal(line, col_nr);
            COMPILER_ASSERT(p.has_value(), "expected uint literal after @ in instruction", file_name, line, line_nr,
                            col_nr);

            expect_space(file_name, line, line_nr, col_nr);
        }

        std::string identifier;
        identifier = expect_directive_or_label_or_inst(file_name, line, line_nr, col_nr);

        if (identifier == ".section")
        {
            curr_addr = 0;
        }
        else if (identifier == ".byte")
        {
            // expect space
            expect_space(file_name, line, line_nr, col_nr);

            // expect an arbitrary number of int literals separated by commas
            while (col_nr < line.size())
            {
                auto byte = expect_uint_literal(line, col_nr);
                COMPILER_ASSERT(byte.has_value(), "expected uint literal after .byte directive", file_name, line,
                                line_nr, col_nr);
                COMPILER_ASSERT(byte.value() <= 0xFF, "byte value must be in range 0-255", file_name, line, line_nr,
                                col_nr);
                curr_addr++;


                if (line[col_nr] == ',')
                {
                    col_nr++;
                    // expect space
                    expect_space(file_name, line, line_nr, col_nr);
                    continue;
                }
                break;
            }
        }
        else if (identifier == ".short")
        {
            // expect space
            expect_space(file_name, line, line_nr, col_nr);

            // expect an arbitrary number of int literals separated by commas
            while (col_nr < line.size())
            {
                auto short_literal = expect_uint_literal(line, col_nr);
                COMPILER_ASSERT(short_literal.has_value(), "expected uint literal after .short directive",
                                file_name, line,
                                line_nr, col_nr);
                COMPILER_ASSERT(short_literal.value() <= 0xFFFF, ".short value must be in range 0-65535", file_name,
                                line,
                                line_nr, col_nr);
                curr_addr += 2;

                if (line[col_nr] == ',')
                {
                    col_nr++;
                    // expect space
                    expect_space(file_name, line, line_nr, col_nr);
                    continue;
                }
                break;
            }
        }
        else if (identifier == ".word")
        {
            // expect space
            expect_space(file_name, line, line_nr, col_nr);

            // expect an arbitrary number of int literals separated by commas
            while (col_nr < line.size())
            {
                // check if "index@" is the next literal, treat as a section index literal
                if (line[col_nr] == 'i')
                {
                    auto idx_literal = expect_identifier(line, col_nr);
                    COMPILER_ASSERT(idx_literal == "index",
                                    "expected index after .word directive, use index@ to reference a section index",
                                    file_name, line, line_nr, col_nr);
                    // expect @
                    COMPILER_ASSERT(line[col_nr] == '@', "expected @ after index in .word directive", file_name,
                                    line,
                                    line_nr, col_nr);
                    col_nr++;

                    // expect parenthesis
                    COMPILER_ASSERT(line[col_nr] == '(', "expected parenthesis after @ in .word directive",
                                    file_name,
                                    line, line_nr, col_nr);
                    col_nr++;

                    auto symbol_name = expect_identifier(line, col_nr);

                    COMPILER_ASSERT(!symbol_name.empty(), "expected symbol name after ( in .word directive", file_name,
                                    line, line_nr, col_nr);

                    curr_addr += 4;

                    // expect closing parenthesis
                    COMPILER_ASSERT(line[col_nr] == ')',
                                    "expected closing parenthesis after symbol name in .word directive",
                                    file_name, line, line_nr, col_nr);
                    col_nr++;
                }
                else
                {
                    auto word = expect_uint_literal(line, col_nr);
                    COMPILER_ASSERT(word.has_value(), "expected uint literal after .word directive", file_name,
                                    line,
                                    line_nr, col_nr);
                    COMPILER_ASSERT(word.value() <= 0xFFFFFFFF, "word value must be in range 0-4294967295",
                                    file_name,
                                    line, line_nr, col_nr);
                    curr_addr += 4;
                }

                if (line[col_nr] == ',')
                {
                    col_nr++;
                    // expect space
                    expect_space(file_name, line, line_nr, col_nr);
                    continue;
                }
                break;
            }
        }
        else if (identifier.back() == ':')
        {
            AsmLabel label{};
            label.name = identifier.substr(0, identifier.size() - 1);
            label.section_offset = curr_addr;
            labels.push_back(label);
        }
        else if (identifier[0] != '.')
        {
            // is instruction
            curr_addr += sizeof(SassInstructionDataSm89);
        }
    }

    // actual parsing pass
    stream = std::istringstream(sass_asm);
    line_nr = 0;
    while (true)
    {
        bool is_eof = !std::getline(stream, line);
        line_nr++;

        if (!is_eof)
        {
            if (line.empty())
            {
                continue;
            }

            for (size_t i = 0; i < line.size() - 1; i++)
            {
                // strip everything after comment identifiers
                if (line[i] == '/' && line[i + 1] == '/' || line[i] == ';')
                {
                    line = line.substr(0, i);
                    break;
                }

                // strip everything between /* and */
                if (line[i] == '/' && line[i + 1] == '*')
                {
                    if (size_t end_comment = line.find("*/", i + 2); end_comment != std::string::npos)
                    {
                        line = line.substr(0, i) + line.substr(end_comment + 2);
                        i--;
                    }
                    else
                    {
                        COMPILER_ASSERT(false, "Unclosed comment", file_name, line, line_nr, i);
                    }
                }
            }

            // ignore empty lines
            if (line.empty())
            {
                continue;
            }

            line = sanitize(line);

            if (line.empty())
            {
                continue;
            }

            // resolve label expressions
            for (size_t i = 0; i < line.size() - 1; i++)
            {
                if ((line[i] == '`' && line[i + 1] == '(') ||
                    // paren is only evaluated for labels on .byte, .short, .word directives
                    ((line.find(".byte") == 0 || line.find(".short") == 0 || line.find(".word") == 0 || line.
                            find(".size") == 0) && line[i] != '@'
                        && line[i + 1] == '('))
                {
                    // find closing parenthesis
                    size_t end_expr = line.find(')', i + 2);
                    COMPILER_ASSERT(end_expr != std::string::npos, "Unclosed label expression", file_name, line,
                                    line_nr,
                                    i);

                    std::string expression = line.substr(i + 2, end_expr - i - 2);

                    for (auto& [name, addr] : labels)
                    {
                        // replace all occurrences of label expressions with their addresses
                        while (true)
                        {
                            size_t label_expr_pos = expression.find(name);
                            if (label_expr_pos == std::string::npos)
                            {
                                break;
                            }
                            expression.replace(label_expr_pos, name.size(), std::to_string(addr));
                        }
                    }
                    // evaluate basic arithmetic expressions
                    expression = std::to_string(tinyScriptEval(expression));

                    line = line.substr(0, i + (line[i] == '`' ? 0 : 1)) + expression + line.substr(end_expr + 1);
                }
            }
        }

        int col_nr = 0;

        uint64_t inst_p = -1; // instruction p value
        bool inst_p_negate = false; // instruction p negate

        // check if next character is @, if so, handle p
        if (line[col_nr] == '@')
        {
            col_nr++;

            // optionally expect ! for p_negate
            if (line[col_nr] == '!')
            {
                inst_p_negate = true;
                col_nr++;
            }

            // Expect P
            COMPILER_ASSERT(line[col_nr] == 'P', "expected P after @ in instruction", file_name, line, line_nr, col_nr);
            col_nr++;

            auto p = expect_uint_literal(line, col_nr);
            COMPILER_ASSERT(p.has_value(), "expected uint literal after @ in instruction", file_name, line, line_nr,
                            col_nr);
            inst_p = p.value();
        }

        std::string identifier;
        if (!is_eof)
        {
            identifier = expect_directive_or_label_or_inst(file_name, line, line_nr, col_nr);
        }

        // is directive
        if (identifier == ".headerflags")
        {
            // eg.:
            // .headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM52 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM89)"
            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after directive", file_name, line, line_nr,
                            col_nr);
            col_nr++;
            COMPILER_ASSERT(line[col_nr] == '@', "expected @ after .headerflags directive", file_name, line,
                            line_nr, col_nr);
            col_nr++;

            // expect fake string literal (we parse it as if it had sub-tokens)
            COMPILER_ASSERT(line[col_nr] == '"', "expected string literal after @ in .headerflags directive",
                            file_name,
                            line, line_nr, col_nr);
            col_nr++;

            // handle tex mode
            {
                auto tex_mode = expect_identifier(line, col_nr);
                COMPILER_ASSERT(!tex_mode.empty(),
                                "expected EF_CUDA_TEXMODE identifier after @\" in .headerflags directive",
                                file_name,
                                line, line_nr, col_nr);

                COMPILER_ASSERT(tex_mode == "EF_CUDA_TEXMODE_UNIFIED",
                                "Only EF_CUDA_TEXMODE_UNIFIED is supported by sassasm", file_name, line, line_nr,
                                col_nr - tex_mode.size());

                // EF_CUDA_64BIT_ADDRESS probably means bits 8-16 = 00000101
                writer.set_flags(writer.get_flags() | 0b101 << 8);
            }

            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after texture mode in .headerflags directive",
                            file_name, line, line_nr, col_nr);
            col_nr++;

            // handle address mode
            {
                auto address_mode = expect_identifier(line, col_nr);
                COMPILER_ASSERT(!address_mode.empty(),
                                "expected address mode identifier after texture mode in .headerflags directive",
                                file_name,
                                line, line_nr, col_nr);

                COMPILER_ASSERT(address_mode == "EF_CUDA_64BIT_ADDRESS",
                                "Only EF_CUDA_64BIT_ADDRESS is supported by sassasm",
                                file_name, line, line_nr, col_nr - address_mode.size());
            }

            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after address mode in .headerflags directive",
                            file_name, line, line_nr, col_nr);
            col_nr++;


            // handle sm version
            {
                auto sm_version = expect_identifier(line, col_nr);
                COMPILER_ASSERT(!sm_version.empty(),
                                "expected SM verion identifier after texture mode in .headerflags directive",
                                file_name,
                                line, line_nr, col_nr);

                COMPILER_ASSERT(sm_version.size() > 10 && sm_version.substr(0, 10) == "EF_CUDA_SM",
                                "Expected EF_CUDA_SM followed by a version number", file_name, line, line_nr,
                                col_nr - sm_version.size());

                auto sm_version_nr_str = sm_version.substr(10);
                int sm_version_nr = std::stoi(sm_version_nr_str);

                writer.set_flags(writer.get_flags() | (sm_version_nr & 0xFF) << 16);
            }

            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after SM version in .headerflags directive",
                            file_name, line, line_nr, col_nr);
            col_nr++;

            // handle virtual sm version
            {
                auto virtual_sm_prefix = expect_identifier(line, col_nr);
                COMPILER_ASSERT(!virtual_sm_prefix.empty(),
                                "Expected EF_CUDA_VIRTUAL_SM identifier followed by an EF_CUDA_SM version number in parenthesis",
                                file_name,
                                line, line_nr, col_nr);

                COMPILER_ASSERT(virtual_sm_prefix == "EF_CUDA_VIRTUAL_SM",
                                "Expected EF_CUDA_VIRTUAL_SM followed by an EF_CUDA_SM version number in parenthesis",
                                file_name, line, line_nr, col_nr - virtual_sm_prefix.size());

                // expect parenthesis
                COMPILER_ASSERT(line[col_nr] == '(', "Expected parenthesis after EF_CUDA_VIRTUAL_SM identifier",
                                file_name, line, line_nr, col_nr);
                col_nr++;

                auto virtual_sm_version = expect_identifier(line, col_nr);
                COMPILER_ASSERT(!virtual_sm_version.empty(),
                                "Expected EF_CUDA_VIRTUAL_SM identifier followed by an EF_CUDA_SM version number in parenthesis",
                                file_name,
                                line, line_nr, col_nr);

                COMPILER_ASSERT(virtual_sm_version.size() > 10 && virtual_sm_version.substr(0, 10) == "EF_CUDA_SM",
                                "Expected EF_CUDA_SM followed by a version number", file_name, line, line_nr,
                                col_nr - virtual_sm_version.size());

                auto virtual_sm_version_nr_str = virtual_sm_version.substr(10);
                int virtual_sm_version_nr = std::stoi(virtual_sm_version_nr_str);

                // last 8 bits are virtual sm number in the header flags
                writer.set_flags(writer.get_flags() | (virtual_sm_version_nr & 0xFF));

                COMPILER_ASSERT(line[col_nr] == ')',
                                "Expected closing parenthesis after EF_CUDA_VIRTUAL_SM version",
                                file_name, line, line_nr, col_nr);
                col_nr++;
            }

            // expect closing quote
            COMPILER_ASSERT(line[col_nr] == '"',
                            "expected closing quote after virtual SM version in .headerflags directive",
                            file_name, line, line_nr, col_nr);
            col_nr++;
        }
        else if (identifier == ".elftype")
        {
            // eg.:
            // .elftype	@"ET_EXEC"

            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after directive", file_name, line, line_nr,
                            col_nr);
            col_nr++;

            COMPILER_ASSERT(line[col_nr] == '@', "expected @ after .elftype directive", file_name, line,
                            line_nr, col_nr);
            col_nr++;

            auto elf_type = expect_string_literal(line, col_nr);
            COMPILER_ASSERT(!elf_type.empty(), "expected string literal after @ in .elftype directive", file_name,
                            line, line_nr, col_nr);

            if (elf_type == "ET_EXEC")
            {
                writer.set_type(ELFIO::ET_EXEC);
            }
            else
            {
                COMPILER_ASSERT(false, "Unsupported ELF type", file_name, line, line_nr, col_nr - elf_type.size());
            }

            col_nr++;
        }
        else if (identifier == ".section" || is_eof)
        {
            // is a symbol
            if (current_section_state.symbol_state.create_symbol)
            {
                const auto& symbol_state = current_section_state.symbol_state;
                COMPILER_ASSERT(current_section != nullptr,
                                "Symbol declaration used outside of a section",
                                file_name, line, line_nr, col_nr);
                assert(current_section != nullptr);
                auto symbol_idx = symbols.add_symbol(
                    str_writer,
                    symbol_state.symbol_name.c_str(),
                    0, symbol_state.size == UINT32_MAX ? current_section->get_size() : symbol_state.size,
                    symbol_state.bind,
                    // set later when .section ends
                    symbol_state.type, symbol_state.other,
                    current_section->get_index()
                );
                symbol_indices[symbol_state.symbol_name] = symbol_idx;
                symtab_sec->set_info(symtab_sec->get_info() + 1);

                // populate check unresolved symbols
                {
                    if (auto symbol_it = unresolved_symbols.find(symbol_state.symbol_name); symbol_it !=
                        unresolved_symbols.end())
                    {
                        for (const auto& [section_idx, data_offset] : symbol_it->second)
                        {
                            auto section = writer.sections[section_idx];
                            const char* data = section->get_data();
                            auto data_writable = const_cast<char*>(data + data_offset);
                            auto symbol_addr = reinterpret_cast<uint32_t*>(data_writable);
                            *symbol_addr = symbol_idx;
                        }
                        unresolved_symbols.erase(symbol_it);
                    }
                }

                current_section_state.symbol_state = {};
            }
            if (identifier == ".section")
            {
                // eg.:
                // .section	.text.write_float,"ax",@progbits
                COMPILER_ASSERT(line[col_nr] == ' ', "expected space after directive", file_name, line, line_nr,
                                col_nr);
                col_nr++;

                auto section_name = expect_identifier(line, col_nr);
                COMPILER_ASSERT(!section_name.empty(), "expected identifier after .section directive", file_name,
                                line, line_nr, col_nr);

                COMPILER_ASSERT(line[col_nr] == ',', "expected comma after section name in .section directive",
                                file_name, line, line_nr, col_nr);
                col_nr++;

                // create the section in the ELF file
                auto section = writer.sections.add(section_name);
                // consider all .text. sections as code/program sections
                if (section_name.find(".text.") == 0)
                {
                    std::string function_name = section_name.substr(6);
                    text_sections[function_name] = section;
                    section->set_link(symtab_sec->get_index());

                    // link to the corresponding .nv.constant0.<function_name> section
                    if (auto it = constant_bank_sections.find(function_name); it != constant_bank_sections.end())
                    {
                        auto const0_section = it->second;
                        const0_section->set_info(section->get_index());
                    }

                    // link to the corresponding .nv.info.<function_name> section
                    if (auto it = info_sections.find(function_name); it != info_sections.end())
                    {
                        it->second->set_info(section->get_index());
                    }
                }

                if (section_name.find(".nv.constant0.") == 0)
                {
                    std::string function_name = section_name.substr(14);
                    constant_bank_sections[function_name] = section;
                    section->set_flags(section->get_flags() | ELFIO::SHF_INFO_LINK);

                    // link to the corresponding .text.<function_name> section
                    if (auto it = text_sections.find(function_name); it != text_sections.end())
                    {
                        section->set_info(it->second->get_index());
                    }
                }

                if (section_name.find(".nv.info") == 0)
                {
                    section->set_link(symtab_sec->get_index());
                }

                if (section_name.find(".nv.info.") == 0)
                {
                    std::string function_name = section_name.substr(9);
                    section->set_flags(section->get_flags() | ELFIO::SHF_INFO_LINK);

                    // link to the corresponding .text.<function_name> section
                    if (auto it = text_sections.find(function_name); it != text_sections.end())
                    {
                        section->set_info(it->second->get_index());
                    }
                    info_sections[function_name] = section;
                }
                COMPILER_ASSERT(section->get_address() == 0,
                                "Internal error: section address already set; this should not happen", file_name, "", 0,
                                0);

                auto flags = expect_string_literal(line, col_nr);

                size_t flag_char_idx = 0;

                // build section flags
                for (char c : flags)
                {
                    ELFIO::Elf32_Word flag;
                    switch (c)
                    {
                    case 'w':
                        flag = ELFIO::SHF_WRITE;
                        break;
                    case 'a':
                        flag = ELFIO::SHF_ALLOC;
                        break;
                    case 'x':
                        flag = ELFIO::SHF_EXECINSTR;
                        break;
                    case 'm':
                        flag = ELFIO::SHF_MERGE;
                        break;
                    case 's':
                        flag = ELFIO::SHF_STRINGS;
                        break;
                    case 'i':
                        flag = ELFIO::SHF_INFO_LINK;
                        break;
                    case 'l':
                        flag = ELFIO::SHF_LINK_ORDER;
                        break;
                    case 'o':
                        flag = ELFIO::SHF_OS_NONCONFORMING;
                        break;
                    case 'g':
                        flag = ELFIO::SHF_GROUP;
                        break;
                    case 't':
                        flag = ELFIO::SHF_TLS;
                        break;
                    case 'c':
                        flag = ELFIO::SHF_COMPRESSED;
                        break;
                    default:
                        COMPILER_ASSERT(false, "Unknown section flag '" + std::to_string(c) + "\'", file_name, line,
                                        line_nr, col_nr
                                        - flags.size() - 1 + flag_char_idx);
                    }
                    section->set_flags(section->get_flags() | flag);
                    flag_char_idx++;
                }

                // expect comma
                COMPILER_ASSERT(line[col_nr] == ',', "expected comma after flags in .section directive", file_name,
                                line,
                                line_nr, col_nr);
                col_nr++;

                // expect @
                COMPILER_ASSERT(line[col_nr] == '@', "expected @ after comma in .section directive", file_name, line,
                                line_nr, col_nr);
                col_nr++;

                // section type
                if (auto section_type = expect_string_literal_or_identifier(line, col_nr); section_type == "progbits")
                {
                    section->set_type(ELFIO::SHT_PROGBITS);
                }
                else if (section_type == "SHT_CUDA_INFO")
                {
                    section->set_type(SHT_CUDA_INFO);
                }
                else if (section_type == "SHT_CUDA_CALLGRAPH")
                {
                    section->set_type(SHT_CUDA_CALLGRAPH);
                }
                else if (section_type == "SHT_CUDA_RELOCINFO")
                {
                    section->set_type(SHT_CUDA_RELOCINFO);
                }
                else
                {
                    COMPILER_ASSERT(false, "Unsupported section type", file_name, line, line_nr,
                                    col_nr - section_type.size());
                }
                current_section = section;

                current_section_state = {};
                current_section_state.symbol_state.create_symbol = true;
                current_section_state.symbol_state.symbol_name = section_name;
                current_section_state.symbol_state.bind = ELFIO::STB_LOCAL;
            }
        }
        else if (identifier == ".sectioninfo")
        {
            // eg.:
            // .sectioninfo	@"SHI_REGISTERS=8"
            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after .sectioninfo directive", file_name, line,
                            line_nr,
                            col_nr);
            col_nr++;

            // assert current section is set
            COMPILER_ASSERT(current_section != nullptr, "sectioninfo directive is illegal without current section",
                            file_name, line, line_nr, col_nr);

            // expect @
            COMPILER_ASSERT(line[col_nr] == '@', "expected @ after .sectioninfo directive", file_name, line,
                            line_nr, col_nr);
            col_nr++;

            // expect fake string literal (we parse it as if it had sub-tokens)
            COMPILER_ASSERT(line[col_nr] == '"', "expected string literal after @ in .sectioninfo directive",
                            file_name,
                            line, line_nr, col_nr);
            col_nr++;

            // parse key value pairs
            while (col_nr < line.size())
            {
                auto key = expect_identifier(line, col_nr);
                COMPILER_ASSERT(!key.empty(), "expected key in key value pair in .sectioninfo directive", file_name,
                                line, line_nr, col_nr);

                COMPILER_ASSERT(line[col_nr] == '=',
                                "expected = after key in key value pair in .sectioninfo directive",
                                file_name, line, line_nr, col_nr);
                col_nr++;

                auto value = expect_identifier(line, col_nr);
                COMPILER_ASSERT(!value.empty(), "expected value in key value pair in .sectioninfo directive",
                                file_name,
                                line, line_nr, col_nr);

                assert(current_section != nullptr);
                if (key == "SHI_REGISTERS")
                {
                    // SHI_REGISTER count is encoded in 8 bits << 24 of info
                    current_section->set_info(std::stoi(value) << 24);
                }
                else
                {
                    COMPILER_ASSERT(false, "Unsupported section info key", file_name, line, line_nr,
                                    col_nr - key.size());
                }

                if (line[col_nr] == ',')
                {
                    col_nr++;
                    continue;
                }
                if (line[col_nr] == '"')
                {
                    col_nr++;
                    break;
                }
                COMPILER_ASSERT(
                    false, "expected comma or closing quote after key value pair in .sectioninfo directive",
                    file_name, line, line_nr, col_nr);
            }
        }
        else if (identifier == ".align")
        {
            COMPILER_ASSERT(current_section != nullptr, ".align directive is illegal without current section",
                            file_name, line, line_nr, col_nr);

            // eg.:
            // .align 128
            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after .align directive", file_name, line, line_nr,
                            col_nr);
            col_nr++;

            auto alignment = expect_uint_literal(line, col_nr);
            if (!alignment.has_value())
            {
                COMPILER_ASSERT(false, "expected uint literal after .align directive", file_name, line, line_nr,
                                col_nr);
            }

            if (current_section->get_size() % alignment.value() != 0)
            {
                std::string zeros(alignment.value() - current_section->get_size() % alignment.value(), 0);
                current_section->append_data(zeros);
            }
            current_section->set_addr_align(alignment.value());
        }
        else if (identifier == ".sectionentsize")
        {
            COMPILER_ASSERT(current_section != nullptr, ".sectionentsize directive is illegal without current section",
                            file_name, line, line_nr, col_nr);

            // eg.: .sectionentsize 8
            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after .sectionentsize directive", file_name, line,
                            line_nr, col_nr);
            col_nr++;

            auto entsize = expect_uint_literal(line, col_nr);
            if (!entsize.has_value())
            {
                COMPILER_ASSERT(false, "expected uint literal after .sectionentsize directive", file_name, line,
                                line_nr, col_nr);
            }

            current_section->set_entry_size(entsize.value());
        }
        else if (identifier == ".sectionflags")
        {
            COMPILER_ASSERT(current_section != nullptr, ".sectionflags directive is illegal without current section",
                            file_name, line, line_nr, col_nr);

            // eg.: .sectionflags @""
            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after .sectionflags directive", file_name, line,
                            line_nr, col_nr);
            col_nr++;

            // expect @
            COMPILER_ASSERT(line[col_nr] == '@', "expected @ after .sectionflags directive", file_name, line,
                            line_nr, col_nr);
            col_nr++;

            auto flags = expect_string_literal(line, col_nr);

            size_t flag_char_idx = 0;

            // build section flags
            for (char c : flags)
            {
                ELFIO::Elf32_Word flag;
                switch (c)
                {
                case 'w':
                    flag = ELFIO::SHF_WRITE;
                    break;
                case 'a':
                    flag = ELFIO::SHF_ALLOC;
                    break;
                case 'x':
                    flag = ELFIO::SHF_EXECINSTR;
                    break;
                case 'm':
                    flag = ELFIO::SHF_MERGE;
                    break;
                case 's':
                    flag = ELFIO::SHF_STRINGS;
                    break;
                case 'i':
                    flag = ELFIO::SHF_INFO_LINK;
                    break;
                case 'l':
                    flag = ELFIO::SHF_LINK_ORDER;
                    break;
                case 'o':
                    flag = ELFIO::SHF_OS_NONCONFORMING;
                    break;
                case 'g':
                    flag = ELFIO::SHF_GROUP;
                    break;
                case 't':
                    flag = ELFIO::SHF_TLS;
                    break;
                case 'c':
                    flag = ELFIO::SHF_COMPRESSED;
                    break;
                default:
                    COMPILER_ASSERT(false, "Unknown section flag '" + std::to_string(c) + "\'", file_name, line,
                                    line_nr, col_nr
                                    - flags.size() - 1 + flag_char_idx);
                }
                current_section->set_flags(current_section->get_flags() | flag);
                flag_char_idx++;
            }
            col_nr++;
        }
        else if (identifier == ".size")
        {
            // assert current section is set
            COMPILER_ASSERT(current_section != nullptr, ".size directive is illegal without current section",
                            file_name, line, line_nr, col_nr);

            // eg.: .size write_float,4
            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after .size directive", file_name, line, line_nr,
                            col_nr);
            col_nr++;

            // expect symbol name
            auto symbol_name = expect_identifier(line, col_nr);
            COMPILER_ASSERT(!symbol_name.empty(), "expected identifier after .size directive", file_name, line,
                            line_nr, col_nr);

            // expect comma
            COMPILER_ASSERT(line[col_nr] == ',', "expected comma after symbol name in .size directive", file_name,
                            line, line_nr, col_nr);
            col_nr++;

            ELFIO::Elf_Xword size;
            if (line[col_nr] == 'a')
            {
                // expect "auto" keyword
                auto a = expect_identifier(line, col_nr);
                COMPILER_ASSERT(a == "auto",
                                "expected auto keyword after comma in .size directive if no explicit size is given",
                                file_name, line,
                                line_nr, col_nr - a.size());
                size = UINT32_MAX;
            }
            else
            {
                // expect size
                auto size_opt = expect_uint_literal(line, col_nr);
                COMPILER_ASSERT(size_opt.has_value(), "expected uint literal after symbol name in .size directive",
                                file_name,
                                line, line_nr, col_nr);
                size = *size_opt;
            }
            // assert current section symbol name matches that of directive
            COMPILER_ASSERT(symbol_name == current_section_state.symbol_state.symbol_name,
                            "symbol name in .size directive must refer to the current symbol created by eg. \".global\"",
                            file_name, line, line_nr, col_nr - symbol_name.size());

            current_section_state.symbol_state.size = size;
        }
        else if (identifier == ".global")
        {
            // eg.:
            // .global write_float
            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after .global directive", file_name, line, line_nr,
                            col_nr);
            col_nr++;

            auto symbol_name = expect_identifier(line, col_nr);
            COMPILER_ASSERT(!symbol_name.empty(), "expected identifier after .global directive", file_name, line,
                            line_nr, col_nr);

            // assert current section is set
            COMPILER_ASSERT(current_section != nullptr, "global directive is illegal without current section",
                            file_name, line, line_nr, col_nr);
            assert(current_section != nullptr);

            current_section_state = {};
            current_section_state.symbol_state.create_symbol = true;
            current_section_state.symbol_state.symbol_name = symbol_name;
            current_section_state.symbol_state.bind = ELFIO::STB_GLOBAL;
        }
        else if (identifier == ".type")
        {
            // eg.:
            // .type write_float,@function
            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after .type directive", file_name, line, line_nr,
                            col_nr);
            col_nr++;

            COMPILER_ASSERT(current_section != nullptr, "type directive is illegal without current section",
                            file_name, line, line_nr, col_nr);

            auto symbol_name = expect_identifier(line, col_nr);
            COMPILER_ASSERT(!symbol_name.empty(), "expected identifier after .type directive", file_name, line,
                            line_nr, col_nr);

            COMPILER_ASSERT(line[col_nr] == ',', "expected comma after symbol name in .type directive", file_name,
                            line, line_nr, col_nr);
            col_nr++;

            COMPILER_ASSERT(line[col_nr] == '@', "expected @ after comma in .type directive", file_name, line,
                            line_nr, col_nr);
            col_nr++;

            auto type = expect_identifier(line, col_nr);
            COMPILER_ASSERT(!type.empty(), "expected type after @ in .type directive", file_name, line, line_nr,
                            col_nr);

            if (type == "function")
            {
                current_section_state.symbol_state.type = ELFIO::STT_FUNC;
            }
            else if (type == "object")
            {
                current_section_state.symbol_state.type = ELFIO::STT_OBJECT;
            }
            else if (type == "notype")
            {
                current_section_state.symbol_state.type = ELFIO::STT_NOTYPE;
            }
            else
            {
                COMPILER_ASSERT(false, "Unsupported symbol type", file_name, line, line_nr, col_nr - type.size());
            }
        }
        else if (identifier == ".other")
        {
            // eg.:
            // .other write_float,@"STO_CUDA_ENTRY STV_DEFAULT"
            COMPILER_ASSERT(line[col_nr] == ' ', "expected space after .other directive", file_name, line, line_nr,
                            col_nr);
            col_nr++;

            auto symbol_name = expect_identifier(line, col_nr);
            COMPILER_ASSERT(!symbol_name.empty(), "expected identifier after .other directive", file_name, line,
                            line_nr, col_nr);

            COMPILER_ASSERT(symbol_name == current_section_state.symbol_state.symbol_name,
                            "symbol name in .other directive must refer to the current symbol created by eg. \".global\"",
                            file_name, line, line_nr, col_nr - symbol_name.size());

            COMPILER_ASSERT(line[col_nr] == ',', "expected comma after symbol name in .other directive", file_name,
                            line, line_nr, col_nr);
            col_nr++;

            COMPILER_ASSERT(line[col_nr] == '@', "expected @ after comma in .other directive", file_name, line,
                            line_nr, col_nr);
            col_nr++;

            // expect fake string literal (we parse it as if it had sub-tokens)
            COMPILER_ASSERT(line[col_nr] == '"', "expected string literal after @ in .other directive", file_name,
                            line, line_nr, col_nr);
            col_nr++;

            while (col_nr < line.size())
            {
                auto other = expect_identifier(line, col_nr);
                COMPILER_ASSERT(!other.empty(), "expected other after @ in .other directive", file_name, line,
                                line_nr, col_nr);

                if (other == "STO_CUDA_ENTRY")
                {
                    current_section_state.symbol_state.other |= STO_CUDA_ENTRY;
                }
                else if (other == "STV_DEFAULT")
                {
                    current_section_state.symbol_state.other |= ELFIO::STV_DEFAULT;
                }
                else
                {
                    COMPILER_ASSERT(false, "Unsupported symbol visibility", file_name, line, line_nr,
                                    col_nr - other.size());
                }

                if (line[col_nr] == ' ')
                {
                    col_nr++;
                    continue;
                }
                if (line[col_nr] == '"')
                {
                    col_nr++;
                    break;
                }
                COMPILER_ASSERT(
                    false, "expected comma or closing quote after key value pair in .sectioninfo directive",
                    file_name, line, line_nr, col_nr);
            }
        }
        else if (identifier == ".zero")
        {
            COMPILER_ASSERT(current_section != nullptr, "zero directive is illegal without current section",
                            file_name, line, line_nr, col_nr);

            // expect space
            expect_space(file_name, line, line_nr, col_nr);

            // expect the number of bytes to zero
            auto zero_bytes = expect_uint_literal(line, col_nr);
            COMPILER_ASSERT(zero_bytes.has_value(), "expected uint literal after .zero directive", file_name, line,
                            line_nr, col_nr);
            std::string zeros(zero_bytes.value(), 0);
            current_section->append_data(zeros.c_str(), zero_bytes.value());
        }
        else if (identifier == ".byte")
        {
            COMPILER_ASSERT(current_section != nullptr, ".byte directive is illegal without current section",
                            file_name, line, line_nr, col_nr);

            // expect space
            expect_space(file_name, line, line_nr, col_nr);

            // expect an arbitrary number of int literals separated by commas
            while (col_nr < line.size())
            {
                auto byte = expect_uint_literal(line, col_nr);
                COMPILER_ASSERT(byte.has_value(), "expected uint literal after .byte directive", file_name, line,
                                line_nr, col_nr);
                COMPILER_ASSERT(byte.value() <= 0xFF, "byte value must be in range 0-255", file_name, line, line_nr,
                                col_nr);
                current_section->append_data(reinterpret_cast<const char*>(&byte.value()), 1);


                if (line[col_nr] == ',')
                {
                    col_nr++;
                    // expect space
                    expect_space(file_name, line, line_nr, col_nr);
                    continue;
                }
                break;
            }
        }
        else if (identifier == ".short")
        {
            COMPILER_ASSERT(current_section != nullptr, "short directive is illegal without current section",
                            file_name, line, line_nr, col_nr);

            // expect space
            expect_space(file_name, line, line_nr, col_nr);

            // expect an arbitrary number of int literals separated by commas
            while (col_nr < line.size())
            {
                auto short_literal = expect_uint_literal(line, col_nr);
                COMPILER_ASSERT(short_literal.has_value(), "expected uint literal after .short directive",
                                file_name, line,
                                line_nr, col_nr);
                COMPILER_ASSERT(short_literal.value() <= 0xFFFF, ".short value must be in range 0-65535", file_name,
                                line,
                                line_nr, col_nr);
                current_section->append_data(reinterpret_cast<const char*>(&short_literal.value()), 2);

                if (line[col_nr] == ',')
                {
                    col_nr++;
                    // expect space
                    expect_space(file_name, line, line_nr, col_nr);
                    continue;
                }
                break;
            }
        }
        else if (identifier == ".word")
        {
            COMPILER_ASSERT(current_section != nullptr, ".word directive is illegal without current section",
                            file_name, line, line_nr, col_nr);

            // expect space
            expect_space(file_name, line, line_nr, col_nr);

            // expect an arbitrary number of int literals separated by commas
            while (col_nr < line.size())
            {
                // check if "index@" is the next literal, treat as a section index literal
                if (line[col_nr] == 'i')
                {
                    auto idx_literal = expect_identifier(line, col_nr);
                    COMPILER_ASSERT(idx_literal == "index",
                                    "expected index after .word directive, use index@ to reference a section index",
                                    file_name, line, line_nr, col_nr);
                    // expect @
                    COMPILER_ASSERT(line[col_nr] == '@', "expected @ after index in .word directive", file_name,
                                    line,
                                    line_nr, col_nr);
                    col_nr++;

                    // expect parenthesis
                    COMPILER_ASSERT(line[col_nr] == '(', "expected parenthesis after @ in .word directive",
                                    file_name,
                                    line, line_nr, col_nr);
                    col_nr++;

                    // expect symbol name
                    auto symbol_name = expect_identifier(line, col_nr);
                    if (auto symbol_it = symbol_indices.find(symbol_name); symbol_it == symbol_indices.end())
                    {
                        // symbol cannot be resolved yet, add a placeholder
                        uint32_t zero = 0;
                        unresolved_symbols[symbol_name].push_back(UnresolvedSymbolOccurrence{
                            .section_idx = current_section->get_index(),
                            .data_offset = current_section->get_address() + current_section->get_size()
                        });
                        current_section->append_data(reinterpret_cast<const char*>(&zero), 4);
                    }
                    else
                    {
                        current_section->append_data(reinterpret_cast<const char*>(&symbol_it->second), 4);
                    }

                    // expect closing parenthesis
                    COMPILER_ASSERT(line[col_nr] == ')',
                                    "expected closing parenthesis after symbol name in .word directive",
                                    file_name, line, line_nr, col_nr);
                    col_nr++;
                }
                else
                {
                    auto word = expect_uint_literal(line, col_nr);
                    COMPILER_ASSERT(word.has_value(), "expected uint literal after .word directive", file_name,
                                    line,
                                    line_nr, col_nr);
                    COMPILER_ASSERT(word.value() <= 0xFFFFFFFF, "word value must be in range 0-4294967295",
                                    file_name,
                                    line, line_nr, col_nr);
                    current_section->append_data(reinterpret_cast<const char*>(&word.value()), 4);
                }

                if (line[col_nr] == ',')
                {
                    col_nr++;
                    // expect space
                    expect_space(file_name, line, line_nr, col_nr);
                    continue;
                }
                break;
            }
        }
        else if (identifier.back() == ':')
        {
            // labels are already handled in the first pass
        }
        else if (identifier[0] != '.')
        {
            // is instruction
            COMPILER_ASSERT(current_section != nullptr, "instruction mnemonic is illegal without current section",
                            file_name, line, line_nr, col_nr);
            assert(current_section != nullptr);

            SassInstructionDataSm89 data{};

            // parse instruction
            if (identifier == "MOV")
            {
                expect_space(file_name, line, line_nr, col_nr); // expect space after instruction mnemonic

                MovRegBankSm89 inst{};
                if (inst_p != -1)
                {
                    inst.p = inst_p;
                    inst.p_negate = inst_p_negate;
                }

                // RN
                uint32_t register_ident = expect_register_identifier(file_name, line, line_nr, col_nr); // expect RN
                expect_parameter_delimiter(file_name, line, line_nr, col_nr); // expect comma
                inst.dst_reg = register_ident;

                // C[bank][addr]
                auto [memory_bank, memory_address] = expect_constant_load_expr(file_name, line, line_nr, col_nr);
                inst.src_bank = memory_bank;
                inst.src_addr = memory_address;
                inst.serialize(data);
            }
            else if (identifier.size() >= 4 && identifier.substr(0, 4) == "IMAD")
            {
                expect_space(file_name, line, line_nr, col_nr); // expect space after instruction mnemonic

                ImadU32Sm89 inst{};
                if (inst_p != -1)
                {
                    inst.p = inst_p;
                    inst.p_negate = inst_p_negate;
                }
                if (identifier[4] == '.')
                {
                    if (identifier.size() >= 8)
                    {
                        bool is_mov = false;
                        if (identifier.substr(5, 3) == "MOV")
                        {
                            is_mov = true;
                        }
                        else if (identifier.substr(5, 3) == "U32")
                        {
                            // do nothing, instruction is .U32 by default
                        }
                        else
                        {
                            COMPILER_ASSERT(false, "Unknown qualifier for IMAD instruction", file_name, line, line_nr,
                                            col_nr - identifier.size());
                        }

                        // expect first register (dst)
                        uint32_t reg_dst = expect_register_identifier(file_name, line, line_nr, col_nr);
                        // expect RN
                        expect_parameter_delimiter(file_name, line, line_nr, col_nr); // expect comma
                        inst.dst = reg_dst;

                        // expect second register (src_a)
                        uint32_t reg_a = expect_register_identifier(file_name, line, line_nr, col_nr); // expect RN
                        COMPILER_ASSERT(reg_a == REG_RZ || !is_mov,
                                        "src_a operand of IMAD must be REG_Z when .MOV qualifier is used",
                                        file_name, line,
                                        line_nr, col_nr);
                        expect_parameter_delimiter(file_name, line, line_nr, col_nr); // expect comma
                        inst.src_a = reg_a;

                        // expect third register (src_b)
                        uint32_t reg_b = expect_register_identifier(file_name, line, line_nr, col_nr); // expect RN
                        COMPILER_ASSERT(reg_a == REG_RZ || !is_mov,
                                        "src_a operand of IMAD must be REG_Z when .MOV qualifier is used",
                                        file_name, line,
                                        line_nr, col_nr);
                        expect_parameter_delimiter(file_name, line, line_nr, col_nr); // expect comma
                        inst.src_b = reg_b;

                        // expect constant load expression
                        auto [memory_bank, memory_address] = expect_constant_load_expr(
                            file_name, line, line_nr, col_nr);
                        inst.src_bank = memory_bank;
                        inst.src_addr = memory_address;
                    }
                }
                else
                {
                    COMPILER_ASSERT(false, "IMAD instruction cannot stand alone. Must be qualified by eg. .U32 or .MOV",
                                    file_name, line, line_nr, col_nr);
                }
                inst.serialize(data);
            }
            else if (identifier.size() >= 4 && identifier.substr(0, 4) == "ULDC")
            {
                expect_space(file_name, line, line_nr, col_nr); // expect space after instruction mnemonic
                UldcRegBankSm89 inst{};
                if (inst_p != -1)
                {
                    inst.up = inst_p;
                }
                if (identifier[4] == '.')
                {
                    if (identifier.size() >= 7)
                    {
                        if (auto substr = identifier.substr(0, 7); substr == "ULDC.U8")
                        {
                            inst.dtype = DtypeSm89::U8;
                        }
                        else if (substr == "ULDC.S8")
                        {
                            inst.dtype = DtypeSm89::S8;
                        }
                        else if (substr == "ULDC.64")
                        {
                            inst.dtype = DtypeSm89::U64;
                        }
                    }
                    else if (identifier.size() >= 8)
                    {
                        if (auto substr = identifier.substr(0, 8); substr == "ULDC.U16")
                        {
                            inst.dtype = DtypeSm89::U16;
                        }
                        else if (substr == "ULDC.S16")
                        {
                            inst.dtype = DtypeSm89::S16;
                        }
                        else if (substr == "ULDC.U32")
                        {
                            inst.dtype = DtypeSm89::U32;
                        }
                    }
                }
                else
                {
                    inst.dtype = DtypeSm89::U32;
                }

                // Expect U before RN
                COMPILER_ASSERT(line[col_nr] == 'U', "expected U before register identifier in ULDC instruction",
                                file_name, line, line_nr,
                                col_nr);
                col_nr++;

                // expect first register (dst)
                uint32_t reg_dst = expect_register_identifier(file_name, line, line_nr, col_nr);
                expect_parameter_delimiter(file_name, line, line_nr, col_nr); // expect comma
                inst.dst = reg_dst;

                // expect constant load expression
                auto [memory_bank, memory_address] = expect_constant_load_expr(file_name, line, line_nr, col_nr);
                inst.src_bank = memory_bank;
                inst.src_addr = memory_address;

                inst.serialize(data);
            }
            else if (identifier.size() >= 3 && identifier.substr(0, 3) == "LDG")
            {
                expect_space(file_name, line, line_nr, col_nr); // expect space after instruction mnemonic

                LdgUrSm89 inst{};
                if (inst_p != -1)
                {
                    inst.p = inst_p;
                    inst.p_negate = inst_p_negate;
                }
                if (identifier[3] == '.')
                {
                    size_t i = 4;

                    while (i < identifier.size())
                    {
                        std::string qualifier{};
                        while (i < identifier.size() && identifier[i] != '.')
                        {
                            qualifier += identifier[i];
                            i++;
                        }

                        // check qualifier
                        {
                            if (qualifier == "E")
                            {
                                inst.is_e = true;
                            }
                            // mode
                            else if (qualifier == "EF")
                            {
                                inst.mode = ModeSm89::EF;
                            }
                            else if (qualifier == "EL")
                            {
                                inst.mode = ModeSm89::EL;
                            }
                            else if (qualifier == "LU")
                            {
                                inst.mode = ModeSm89::LU;
                            }
                            else if (qualifier == "EU")
                            {
                                inst.mode = ModeSm89::EU;
                            }
                            else if (qualifier == "NA")
                            {
                                inst.mode = ModeSm89::NA;
                            }
                            // data types
                            else if (qualifier == "U8")
                            {
                                inst.dtype = DtypeSm89::U8;
                            }
                            else if (qualifier == "S8")
                            {
                                inst.dtype = DtypeSm89::S8;
                            }
                            else if (qualifier == "U16")
                            {
                                inst.dtype = DtypeSm89::U16;
                            }
                            else if (qualifier == "S16")
                            {
                                inst.dtype = DtypeSm89::S16;
                            }
                            else if (qualifier == "U32")
                            {
                                inst.dtype = DtypeSm89::U32;
                            }
                            else if (qualifier == "64")
                            {
                                inst.dtype = DtypeSm89::U64;
                            }
                            else
                            {
                                COMPILER_ASSERT(
                                    false,
                                    "Unknown qualifier for LDG instruction",
                                    file_name, line, line_nr, col_nr - qualifier.size());
                            }
                        }
                    }
                }
                // expect p_dst predicate
                if (line[col_nr] == 'P')
                {
                    uint32_t p_dst = expect_predicate_identifier(file_name, line, line_nr, col_nr);
                    inst.pred_dst = p_dst;
                    col_nr++;
                }

                // expect first register (dst)
                uint32_t reg_dst = expect_register_identifier(file_name, line, line_nr, col_nr);
                expect_parameter_delimiter(file_name, line, line_nr, col_nr); // expect comma
                inst.dst = reg_dst;

                // expect src ur expression
                auto [base_register, offset_register, offset_imm, base_reg_64_bit]
                    = expect_fused_register_offset_expr(file_name, line, line_nr, col_nr);
                inst.src = base_register;
                if (offset_register != -1)
                {
                    inst.src_offreg = offset_register;
                }
                inst.src_offimm = offset_imm;
                if (offset_register == -1 && offset_imm == 0)
                {
                    inst.no_regoffcalc = true;
                }
                inst.src_is64bit = base_reg_64_bit;

                inst.serialize(data);
            }
            else if (identifier.size() >= 3 && identifier.substr(0, 3) == "STG")
            {
                expect_space(file_name, line, line_nr, col_nr); // expect space after instruction mnemonic

                StgUrSm89 inst{};
                if (inst_p != -1)
                {
                    inst.p = inst_p;
                    inst.p_negate = inst_p_negate;
                }
                if (identifier[3] == '.')
                {
                    size_t i = 4;
                    while (i < identifier.size())
                    {
                        std::string qualifier{};
                        while (i < identifier.size() && identifier[i] != '.')
                        {
                            qualifier += identifier[i];
                            i++;
                        }
                        i++;

                        // check qualifier
                        {
                            if (qualifier == "E")
                            {
                                inst.is_e = true;
                            }
                            // mode
                            else if (qualifier == "EF")
                            {
                                inst.mode = ModeSm89::EF;
                            }
                            else if (qualifier == "EL")
                            {
                                inst.mode = ModeSm89::EL;
                            }
                            else if (qualifier == "LU")
                            {
                                inst.mode = ModeSm89::LU;
                            }
                            else if (qualifier == "EU")
                            {
                                inst.mode = ModeSm89::EU;
                            }
                            else if (qualifier == "NA")
                            {
                                inst.mode = ModeSm89::NA;
                            }
                            // data types
                            else if (qualifier == "U8")
                            {
                                inst.dtype = DtypeSm89::U8;
                            }
                            else if (qualifier == "S8")
                            {
                                inst.dtype = DtypeSm89::S8;
                            }
                            else if (qualifier == "U16")
                            {
                                inst.dtype = DtypeSm89::U16;
                            }
                            else if (qualifier == "S16")
                            {
                                inst.dtype = DtypeSm89::S16;
                            }
                            else if (qualifier == "U32")
                            {
                                inst.dtype = DtypeSm89::U32;
                            }
                            else if (qualifier == "64")
                            {
                                inst.dtype = DtypeSm89::U64;
                            }
                            else
                            {
                                COMPILER_ASSERT(
                                    false,
                                    "Unknown qualifier for STG instruction",
                                    file_name, line, line_nr, col_nr - qualifier.size());
                            }
                        }
                    }
                }

                // expect dst ur expression
                auto [base_register, offset_register, offset_imm, base_reg_64_bit]
                    = expect_fused_register_offset_expr(file_name, line, line_nr, col_nr);
                inst.dst = base_register;
                if (offset_register != -1)
                {
                    inst.dst_offreg = offset_register;
                }
                inst.dst_offimm = offset_imm;
                if (offset_register == -1 && offset_imm == 0)
                {
                    inst.no_regoffcalc = true;
                }
                COMPILER_ASSERT(inst.dst_is64bit, "STG dst must be 64-bit, as it is an address", file_name, line,
                                line_nr, col_nr);

                // expect src register
                expect_parameter_delimiter(file_name, line, line_nr, col_nr); // expect comma
                uint32_t reg_src = expect_register_identifier(file_name, line, line_nr, col_nr);
                inst.src = reg_src;

                inst.serialize(data);
            }
            else if (identifier == "S2R")
            {
                S2rSm89 inst{};
                if (inst_p != -1)
                {
                    inst.p = inst_p;
                    inst.p_negate = inst_p_negate;
                }
                expect_space(file_name, line, line_nr, col_nr); // expect space after instruction mnemonic

                // RN
                uint32_t register_ident = expect_register_identifier(file_name, line, line_nr, col_nr); // expect RN
                expect_parameter_delimiter(file_name, line, line_nr, col_nr); // expect comma
                inst.dst = register_ident;

                // SR_...
                auto special_register = expect_special_register(line, col_nr);
                inst.src = special_register;

                inst.serialize(data);
            }
            else if (identifier == "BRA")
            {
                BraSm89 inst{};
                if (inst_p != -1)
                {
                    inst.p = inst_p;
                    inst.p_negate = inst_p_negate;
                }
                expect_space(file_name, line, line_nr, col_nr); // expect space after instruction mnemonic

                // expect address literal
                auto address = expect_address_literal(file_name, line, line_nr, col_nr);
                inst.off_imm = static_cast<int64_t>(current_section->get_size()) - static_cast<int64_t>(sizeof(
                    SassInstructionDataSm89)) - address;
                inst.serialize(data);
            }
            else if (identifier == "EXIT")
            {
                ExitSm89 inst{};
                if (inst_p != -1)
                {
                    inst.p = inst_p;
                    inst.p_negate = inst_p_negate;
                }
                inst.serialize(data);
            }
            else if (identifier == "NOP")
            {
                NopSm89 inst{};
                if (inst_p != -1)
                {
                    inst.p = inst_p;
                    inst.p_negate = inst_p_negate;
                }
                inst.serialize(data);
            }
            else
            {
                COMPILER_ASSERT(false, "Unknown instruction mnemonic", file_name, line, line_nr,
                                col_nr - identifier.size());
            }

            // expect_statement_end(file_name, line, line_nr, col_nr);

            current_section->append_data(reinterpret_cast<const char*>(data.data), sizeof(data));
        }
        else if (identifier[0] == '.')
        {
            std::cout << "Encountered unknown directive: " << identifier << std::endl;
        }

        if (is_eof)
        {
            break;
        }
    }
    {
        // create LOAD segments for constant banks & segment mappings
        for (auto& constant_bank_section : constant_bank_sections | std::views::values)
        {
            ELFIO::segment* load_seg = writer.segments.add();
            load_seg->set_type(ELFIO::PT_LOAD);
            load_seg->set_flags(ELFIO::PF_R | ELFIO::PF_X);

            // we use zero as a placeholder for the actual section index because if we referenced the
            // actual sections already, the ordering of the ELF-file would be wrong because of ELFIO rules.
            // We therefore crudely patch this afterward.
            // This code only makes sense in context of what happens in patch_elf()
            load_seg->add_section_index(0, constant_bank_section->get_addr_align());
        }
    }
    writer.save("out.elf");

    // Patch everything with libelf that ELFIO can't do for some godforsaken reason
    patch_elf("out.elf");
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        return 0;
    }

    const char* file_name = argv[1];

    std::string sass_asm{};
    // read file
    {
        std::ifstream file(file_name);
        if (!file.is_open())
        {
            std::cerr << "Error: could not open file " << file_name << std::endl;
            return 1;
        }
        std::string line;
        while (std::getline(file, line))
        {
            sass_asm += line + "\n";
        }
    }

    const std::string arch = "sm_89";
    assemble_file(file_name, sass_asm, arch);
}
