from elftools.elf.elffile import ELFFile

def print_section_indices_and_shnum(file_path):
    with open(file_path, 'rb') as f:
        elf = ELFFile(f)

        # Iterate over the sections
        for section_index, section in enumerate(elf.iter_sections()):
            print(f"Section Index: {section_index}, shnum: {section['sh_name']}")

# Provide the path to your ELF file here
if __name__ == '__main__':
    elf_file_path = 'out.elf'
    print_section_indices_and_shnum(elf_file_path)
