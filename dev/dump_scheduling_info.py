#!/usr/bin/env python3

import sys
import struct
import subprocess
from elftools.elf.elffile import ELFFile
import re

def read_bits(offset, length, src):
    """
    Extracts a bitfield from the source list of 64-bit words.

    :param offset: Bit offset from the start.
    :param length: Number of bits to extract.
    :param src: List of 64-bit integers representing the source bits.
    :return: Extracted integer value.
    """
    src_bit_size = 64
    if length == 64:
        value_mask = 0xFFFFFFFFFFFFFFFF
    else:
        value_mask = (~(0xFFFFFFFFFFFFFFFF << length)) & 0xFFFFFFFFFFFFFFFF

    code_word_offset = offset // src_bit_size
    bit_in_word_offset = offset % src_bit_size
    bit_shift = bit_in_word_offset

    if bit_shift + length <= src_bit_size:
        maskHWord = (value_mask << bit_shift) & 0xFFFFFFFFFFFFFFFF
        valueHWord = src[code_word_offset] & maskHWord
        return (valueHWord >> bit_shift) & value_mask
    else:
        # Field spans two words
        bits_in_first_word = src_bit_size - bit_shift
        bits_in_second_word = length - bits_in_first_word

        mask_hi_word = (value_mask >> bits_in_second_word) & 0xFFFFFFFFFFFFFFFF
        mask_lo_word = (value_mask << bit_shift) & 0xFFFFFFFFFFFFFFFF

        value_lo_word = src[code_word_offset] & mask_lo_word
        value_hi_word = src[code_word_offset + 1] & mask_hi_word

        combined = ((value_hi_word << bits_in_first_word) | (value_lo_word >> bit_shift)) & value_mask
        return combined

def decode_scheduling_info(scheduling_info):
    """
    Decodes the 21-bit scheduling information into its respective fields.

    :param scheduling_info: A 21-bit integer containing the scheduling information.
    :return: A dictionary with the decoded fields.
    """
    # Reuse (4 bits)
    reuse = (scheduling_info >> 17) & 0xF
    # b-mask (6 bits)
    b_mask = (scheduling_info >> 11) & 0x3F
    # w-bar (3 bits)
    w_bar = (scheduling_info >> 8) & 0x7
    # r-bar (3 bits)
    r_bar = (scheduling_info >> 5) & 0x7
    # y (1 bit)
    y = (scheduling_info >> 4) & 0x1
    # stall (4 bits)
    stall = scheduling_info & 0xF

    return {
        'reuse': reuse,
        'b_mask': b_mask,
        'w_bar': w_bar,
        'r_bar': r_bar,
        'y': y,
        'stall': stall
    }

def invoke_nvdisasm(cubin_path):
    """
    Invokes nvdisasm on the provided cubin file and captures the disassembly output.

    :param cubin_path: Path to the .cubin file.
    :return: String containing the disassembly output.
    """
    try:
        result = subprocess.run(
            ['nvdisasm', cubin_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        if result.returncode != 0:
            print(f"nvdisasm failed with error code {result.returncode}:")
            print(result.stderr)
            sys.exit(1)
        return result.stdout
    except FileNotFoundError:
        print("Error: nvdisasm not found. Please ensure it is installed and in your PATH.")
        sys.exit(1)

def parse_disassembly(disassembly_text):
    """
    Parses the disassembly output and builds a mapping from instruction offsets to instruction lines.

    :param disassembly_text: String containing the disassembly output.
    :return: Dictionary mapping instruction offsets (integers) to instruction lines.
    """
    disasm_mapping = {}

    lines = disassembly_text.splitlines()
    current_section = None
    for line in lines:
        line = line.strip()
        # Check for section headers
        section_match = re.match(r'^//\s+(\.text\..+)$', line)
        if section_match:
            current_section = section_match.group(1)
            continue

        # Match instruction lines
        match = re.match(r'/\*([0-9a-fA-F]+)\*/\s+(.*?);$', line)
        if match:
            offset_hex = match.group(1)
            instruction = match.group(2).strip()
            offset = int(offset_hex, 16)
            disasm_mapping[offset] = instruction
    return disasm_mapping

def dump_scheduling_info(cubin_path, disasm_mapping):
    """
    Reads a .cubin ELF file and dumps scheduling information of instructions
    contained in any section starting with .text, including instruction mnemonics.

    :param cubin_path: Path to the .cubin file.
    :param disasm_mapping: Dictionary mapping instruction offsets to instruction lines.
    """
    try:
        with open(cubin_path, 'rb') as f:
            elffile = ELFFile(f)

            # Iterate over all sections
            for section in elffile.iter_sections():
                if section.name.startswith('.text'):
                    section_data = section.data()
                    section_size = section['sh_size']
                    num_instructions = section_size // 16  # Each instruction is 16 bytes

                    print(f"Section: {section.name}")
                    print(f"Number of instructions: {num_instructions}")

                    for i in range(num_instructions):
                        instr_offset = section['sh_addr'] + i * 16
                        instr_bytes = section_data[i * 16:(i + 1) * 16]

                        if len(instr_bytes) < 16:
                            print(f"Warning: Incomplete instruction at offset {instr_offset}. Skipping.")
                            continue

                        # Unpack as two little-endian unsigned 64-bit integers
                        low_word, high_word = struct.unpack('<QQ', instr_bytes)
                        src = [low_word, high_word]

                        # Extract 21 bits starting at bit offset 105
                        scheduling_info = read_bits(105, 21, src)

                        # Decode the scheduling information
                        decoded_info = decode_scheduling_info(scheduling_info)

                        # Get the instruction mnemonic from disassembly
                        instruction_line = disasm_mapping.get(instr_offset - section['sh_addr'], "Unknown instruction")

                        # Print the decoded fields
                        print(f"Instruction {i} at offset {instr_offset:#06x}:")
                        print(f"  Instruction: {instruction_line}")
                        print(f"  Reuse: {decoded_info['reuse']}")
                        print(f"  b-mask: {decoded_info['b_mask']}")
                        print(f"  w-bar: {decoded_info['w_bar']}")
                        print(f"  r-bar: {decoded_info['r_bar']}")
                        print(f"  y: {decoded_info['y']}")
                        print(f"  Stall: {decoded_info['stall']}\n")

    except FileNotFoundError:
        print(f"Error: File '{cubin_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def print_usage():
    print("Usage: dump_scheduling_info.py <path_to_cubin_file>")

def main():
    if len(sys.argv) != 2:
        print("Error: Incorrect number of arguments.")
        print_usage()
        sys.exit(1)

    cubin_path = sys.argv[1]

    # Invoke nvdisasm and parse its output
    disassembly_text = invoke_nvdisasm(cubin_path)
    disasm_mapping = parse_disassembly(disassembly_text)

    dump_scheduling_info(cubin_path, disasm_mapping)

if __name__ == "__main__":
    main()
