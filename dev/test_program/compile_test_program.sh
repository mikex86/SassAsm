nvcc -ptx -arch=sm_89 test_program.cu -o test_program.ptx
ptxas -arch=sm_89 test_program.ptx -o test_program.cubin
nvdisasm test_program.cubin > test_program.asm