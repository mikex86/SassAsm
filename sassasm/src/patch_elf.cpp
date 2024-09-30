#include "patch_elf.h"

#include <iostream>
#include <fstream>
#include <elfio/elfio.hpp>


void patch_elf(const std::string& filepath)
{
    // Create an instance of the ELFIO reader
    ELFIO::elfio reader;

    // Load the ELF file
    if (!reader.load(filepath))
    {
        std::cerr << "Failed to load ELF file: " << filepath << std::endl;
        return;
    }

    // Iterate over segments to find the one containing the section
    const ELFIO::segment* target_segment = nullptr;
    const ELFIO::Elf_Xword seg_num = reader.segments.size();

    for (ELFIO::Elf_Xword i = 0; i < seg_num; ++i)
    {
        if (const ELFIO::segment* seg = reader.segments[i]; seg->get_type() == ELFIO::PT_LOAD)
        {
            target_segment = seg;
            break;
        }
    }

    const ELFIO::section* target_section = nullptr;
    const ELFIO::Elf_Half sec_num = reader.sections.size();
    for (ELFIO::Elf_Half i = 0; i < sec_num; ++i)
    {
        // iterate over sections to find section that starts with .nv.constant0.
        if (ELFIO::section* sec = reader.sections[i]; sec->get_name().find(".nv.constant0.") == 0)
        {
            target_section = sec;
        }
    }


    std::fstream elf_file(filepath, std::ios::in | std::ios::out | std::ios::binary);
    if (!elf_file)
    {
        std::cerr << "Failed to open ELF file for writing: " << filepath << std::endl;
        return;
    }

    if (target_segment != nullptr && target_section != nullptr)
    {
        // find parent text section
        ELFIO::section* parent_text_section = nullptr;
        for (ELFIO::Elf_Half i = 0; i < sec_num; ++i)
        {
            if (ELFIO::section* sec = reader.sections[i]; sec->get_name().find(".text.") == 0
                // len(".text.") = 6
                // len(".nv.constant0.") = 14
                && sec->get_name().substr(6).find(target_section->get_name().substr(14)) == 0)
            {
                parent_text_section = sec;
            }
        }

        if (parent_text_section == nullptr)
        {
            std::cerr << "Failed to find parent text section for section: " << target_section->get_name() << std::endl;
            return;
        }

        auto new_p_offset = target_section->get_offset();
        auto p_offset_field_file_offset = reader.get_segments_offset() +
            target_segment->get_index() * reader.get_segment_entry_size() +
            offsetof(ELFIO::Elf64_Phdr, p_offset);

        auto p_filesz_field_file_offset = reader.get_segments_offset() +
            target_segment->get_index() * reader.get_segment_entry_size() +
            offsetof(ELFIO::Elf64_Phdr, p_filesz);

        auto p_memsz_field_file_offset = reader.get_segments_offset() +
            target_segment->get_index() * reader.get_segment_entry_size() +
            offsetof(ELFIO::Elf64_Phdr, p_memsz);

        auto p_align_field_file_offset = reader.get_segments_offset() +
            target_segment->get_index() * reader.get_segment_entry_size() +
            offsetof(ELFIO::Elf64_Phdr, p_align);

        if (bool is_64bit = reader.get_class() == ELFIO::ELFCLASS64; !is_64bit)
        {
            std::cout << "Only 64-bit ELF files are supported" << std::endl;
            return;
        }

        // seek to 0x14 and replace the byte with 0x7E
        {
            elf_file.seekp(0x14);
            char byte_to_write = 0x7E;
            elf_file.write(&byte_to_write, 1);
        }

        // Seek to the position of the p_offset field
        elf_file.seekp(p_offset_field_file_offset);
        ELFIO::Elf64_Off p_offset_to_write = new_p_offset;
        elf_file.write(reinterpret_cast<const char*>(&p_offset_to_write), sizeof(ELFIO::Elf64_Off));

        // set load segment size
        {
            ELFIO::Elf_Xword load_size = parent_text_section->get_offset() + parent_text_section->get_size() - target_section->get_offset();

            // Seek to the position of the p_filesz field
            elf_file.seekp(p_filesz_field_file_offset);
            elf_file.write(reinterpret_cast<const char*>(&load_size), sizeof(ELFIO::Elf_Xword));

            // Seek to the position of the p_memsz field
            elf_file.seekp(p_memsz_field_file_offset);
            elf_file.write(reinterpret_cast<const char*>(&load_size), sizeof(ELFIO::Elf_Xword));
        }

        // Seek to the position of the p_align field
        elf_file.seekp(p_align_field_file_offset);
        ELFIO::Elf_Xword p_align_to_write = 8;

        elf_file.write(reinterpret_cast<const char*>(&p_align_to_write), sizeof(ELFIO::Elf_Xword));
    }
}
