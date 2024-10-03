	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM89)"
	.elftype	@"ET_EXEC"

//--------------------- .text.fibonacciKernel     --------------------------
	.section	.text.fibonacciKernel,"ax",@progbits
	.sectioninfo	@"SHI_REGISTERS=8"
	.align	128
        .global         fibonacciKernel
        .type           fibonacciKernel,@function
        .size           fibonacciKernel,(.L_x_8 - fibonacciKernel)
        .other          fibonacciKernel,@"STO_CUDA_ENTRY STV_DEFAULT"
fibonacciKernel:
.text.fibonacciKernel:
        ISETP.NE.AND P0, PT, R0, RZ, PT
        EXIT
.END:
    BRA `(.END)
    NOP
