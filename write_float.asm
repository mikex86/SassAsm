//--------------------- elf header --------------------------
.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM89)"
.elftype	@"ET_EXEC"

//--------------------- .nv.info                  --------------------------
.section        .nv.info,"",@"SHT_CUDA_INFO"
        .align 4

        //----- nvinfo : EIATTR_REGCOUNT
        .align 4
            .byte   0x04, 0x2f
            .short  8
        .align 4
            .word   index@(write_float)
            .word   0x00000008

        //----- nvinfo : EIATTR_FRAME_SIZE
        .align 4
            .byte   0x04, 0x11
            .short  8
        .align 4
            .word   index@(write_float)
            .word   0x00000000

        //----- nvinfo : EIATTR_MIN_STACK_SIZE
        .align 4
            .byte   0x04, 0x12
            .short  8
        .align  4
            .word   index@(write_float)
            .word   0x00000000

//--------------------- .nv.info.write_float      --------------------------
        .section        .nv.info.write_float,"",@"SHT_CUDA_INFO"
        .sectionflags   @""
        .align  4


        //----- nvinfo : EIATTR_CUDA_API_VERSION
        .align          4
        .byte   0x04, 0x37
        .short  (.L_7 - .L_6)
.L_6:
        .word   0x0000007e


        //----- nvinfo : EIATTR_PARAM_CBANK
        .align          4
.L_7:
        .byte   0x04, 0x0a
        .short  (.L_9 - .L_8)
        .align          4
.L_8:
        .word   index@(.nv.constant0.write_float)
        .short  0x0160
        .short  0x0010


        //----- nvinfo : EIATTR_CBANK_PARAM_SIZE
        .align          4
.L_9:
        .byte   0x03, 0x19
        .short  0x0010


        //----- nvinfo : EIATTR_KPARAM_INFO
        .align          4
        .byte   0x04, 0x17
        .short  (.L_11 - .L_10)
.L_10:
        .word   0x00000000
        .short  0x0001
        .short  0x0008
        .byte   0x00, 0xf0, 0x21, 0x00


        //----- nvinfo : EIATTR_KPARAM_INFO
        .align          4
.L_11:
        .byte   0x04, 0x17
        .short  (.L_13 - .L_12)
.L_12:
        .word   0x00000000
        .short  0x0000
        .short  0x0000
        .byte   0x00, 0xf0, 0x21, 0x00


        //----- nvinfo : EIATTR_MAXREG_COUNT
        .align          4
.L_13:
        .byte   0x03, 0x1b
        .short  0x00ff


        //----- nvinfo : EIATTR_EXIT_INSTR_OFFSETS
        .align          4
        .byte   0x04, 0x1c
        .short  (.L_15 - .L_14)
.L_14:
        .word   0x00000080
.L_15:


//--------------------- .nv.callgraph             --------------------------
.section	.nv.callgraph,"",@"SHT_CUDA_CALLGRAPH"
	.align	4
	.sectionentsize	8
	.align		4
        .word	0x00000000
	.align		4
        .word	0xffffffff
	.align		4
        .word	0x00000000
	.align		4
        .word	0xfffffffe
	.align		4
        .word	0x00000000
	.align		4
        .word	0xfffffffd
	.align		4
        .word	0x00000000
	.align		4
        .word	0xfffffffc

//--------------------- .nv.constant0.write_float --------------------------
.section        .nv.constant0.write_float,"a",@progbits
        .sectionflags   @""
        .align  4
.nv.constant0.write_float:
        .zero           368

//--------------------- .text.write_float         --------------------------
.section	.text.write_float,"ax",@progbits
	.sectioninfo	@"SHI_REGISTERS=8"
	.align	128
        .global         write_float
        .type           write_float,@function
        .size           write_float,auto
        .other          write_float,@"STO_CUDA_ENTRY STV_DEFAULT"
write_float:
.text.write_float:
    MOV R1, c[0x0][0x28]
    IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x16c]
    MOV R2, c[0x0][0x168]
    ULDC.64 UR4, c[0x0][0x118]
    LDG.E R3, [R2.64]
    MOV R4, c[0x0][0x160]
    IMAD.MOV.U32 R5, RZ, RZ, c[0x0][0x164]
    STG.E [R4.64], R3
    EXIT
.LOOP:
    BRA `(.LOOP)
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP
    NOP