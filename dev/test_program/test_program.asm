	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM89)"
	.elftype	@"ET_REL"


//--------------------- .nv.info                  --------------------------
	.section	.nv.info,"",@"SHT_CUDA_INFO"
	.align	4


	//----- nvinfo : EIATTR_REGCOUNT
	.align		4
        /*0000*/ 	.byte	0x04, 0x2f
        /*0002*/ 	.short	(.L_1 - .L_0)
	.align		4
.L_0:
        /*0004*/ 	.word	index@(fibonacciKernel)
        /*0008*/ 	.word	0x00000008


	//----- nvinfo : EIATTR_FRAME_SIZE
	.align		4
.L_1:
        /*000c*/ 	.byte	0x04, 0x11
        /*000e*/ 	.short	(.L_3 - .L_2)
	.align		4
.L_2:
        /*0010*/ 	.word	index@(fibonacciKernel)
        /*0014*/ 	.word	0x00000000


	//----- nvinfo : EIATTR_MIN_STACK_SIZE
	.align		4
.L_3:
        /*0018*/ 	.byte	0x04, 0x12
        /*001a*/ 	.short	(.L_5 - .L_4)
	.align		4
.L_4:
        /*001c*/ 	.word	index@(fibonacciKernel)
        /*0020*/ 	.word	0x00000000
.L_5:


//--------------------- .nv.info.fibonacciKernel  --------------------------
	.section	.nv.info.fibonacciKernel,"",@"SHT_CUDA_INFO"
	.sectionflags	@""
	.align	4


	//----- nvinfo : EIATTR_CUDA_API_VERSION
	.align		4
        /*0000*/ 	.byte	0x04, 0x37
        /*0002*/ 	.short	(.L_7 - .L_6)
.L_6:
        /*0004*/ 	.word	0x0000007e


	//----- nvinfo : EIATTR_PARAM_CBANK
	.align		4
.L_7:
        /*0008*/ 	.byte	0x04, 0x0a
        /*000a*/ 	.short	(.L_9 - .L_8)
	.align		4
.L_8:
        /*000c*/ 	.word	index@(.nv.constant0.fibonacciKernel)
        /*0010*/ 	.short	0x0160
        /*0012*/ 	.short	0x0010


	//----- nvinfo : EIATTR_CBANK_PARAM_SIZE
	.align		4
.L_9:
        /*0014*/ 	.byte	0x03, 0x19
        /*0016*/ 	.short	0x0010


	//----- nvinfo : EIATTR_KPARAM_INFO
	.align		4
        /*0018*/ 	.byte	0x04, 0x17
        /*001a*/ 	.short	(.L_11 - .L_10)
.L_10:
        /*001c*/ 	.word	0x00000000
        /*0020*/ 	.short	0x0001
        /*0022*/ 	.short	0x0008
        /*0024*/ 	.byte	0x00, 0xf0, 0x21, 0x00


	//----- nvinfo : EIATTR_KPARAM_INFO
	.align		4
.L_11:
        /*0028*/ 	.byte	0x04, 0x17
        /*002a*/ 	.short	(.L_13 - .L_12)
.L_12:
        /*002c*/ 	.word	0x00000000
        /*0030*/ 	.short	0x0000
        /*0032*/ 	.short	0x0000
        /*0034*/ 	.byte	0x00, 0xf0, 0x21, 0x00


	//----- nvinfo : EIATTR_MAXREG_COUNT
	.align		4
.L_13:
        /*0038*/ 	.byte	0x03, 0x1b
        /*003a*/ 	.short	0x00ff


	//----- nvinfo : EIATTR_EXIT_INSTR_OFFSETS
	.align		4
        /*003c*/ 	.byte	0x04, 0x1c
        /*003e*/ 	.short	(.L_15 - .L_14)


	//   ....[0]....
.L_14:
        /*0040*/ 	.word	0x00000030


	//   ....[1]....
        /*0044*/ 	.word	0x00000500
.L_15:


//--------------------- .nv.callgraph             --------------------------
	.section	.nv.callgraph,"",@"SHT_CUDA_CALLGRAPH"
	.align	4
	.sectionentsize	8
	.align		4
        /*0000*/ 	.word	0x00000000
	.align		4
        /*0004*/ 	.word	0xffffffff
	.align		4
        /*0008*/ 	.word	0x00000000
	.align		4
        /*000c*/ 	.word	0xfffffffe
	.align		4
        /*0010*/ 	.word	0x00000000
	.align		4
        /*0014*/ 	.word	0xfffffffd
	.align		4
        /*0018*/ 	.word	0x00000000
	.align		4
        /*001c*/ 	.word	0xfffffffc

//--------------------- .nv.constant0.fibonacciKernel --------------------------
	.section	.nv.constant0.fibonacciKernel,"a",@progbits
	.sectionflags	@""
	.align	4
.nv.constant0.fibonacciKernel:
	.zero		368


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
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   S2R R0, SR_TID.X ;
        /*0020*/                   ISETP.NE.AND P0, PT, R0, RZ, PT ;
        /*0030*/               @P0 EXIT ;
        /*0040*/                   IMAD.MOV.U32 R2, RZ, RZ, c[0x0][0x168] ;
        /*0050*/                   ULDC.64 UR4, c[0x0][0x118] ;
        /*0060*/                   IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x16c] ;
        /*0070*/                   LDG.E R2, [R2.64] ;
        /*0080*/                   ISETP.GE.AND P0, PT, R2, 0x2, PT ;
        /*0090*/                   MOV R5, R2 ;
        /*00a0*/              @!P0 BRA `(.L_x_0) ;
        /*00b0*/                   IADD3 R0, R2, -0x2, RZ ;
        /*00c0*/                   IMAD.MOV.U32 R5, RZ, RZ, 0x1 ;
        /*00d0*/                   IMAD.MOV.U32 R4, RZ, RZ, RZ ;
        /*00e0*/                   ISETP.GE.U32.AND P0, PT, R0, 0x3, PT ;
        /*00f0*/                   IADD3 R0, R2, -0x1, RZ ;
        /*0100*/                   LOP3.LUT R3, R0, 0x3, RZ, 0xc0, !PT ;
        /*0110*/              @!P0 BRA `(.L_x_1) ;
        /*0120*/                   IMAD.IADD R2, R2, 0x1, -R3 ;
        /*0130*/                   MOV R5, 0x1 ;
        /*0140*/                   IMAD.MOV.U32 R4, RZ, RZ, RZ ;
        /*0150*/                   ISETP.GT.AND P0, PT, R2, 0x1, PT ;
        /*0160*/              @!P0 BRA `(.L_x_2) ;
        /*0170*/                   IADD3 R0, R2, -0x1, RZ ;
        /*0180*/                   PLOP3.LUT P0, PT, PT, PT, PT, 0x80, 0x0 ;
        /*0190*/                   ISETP.GT.AND P1, PT, R0, 0xc, PT ;
        /*01a0*/              @!P1 BRA `(.L_x_3) ;
        /*01b0*/                   PLOP3.LUT P0, PT, PT, PT, PT, 0x8, 0x0 ;
.L_x_4:
        /*01c0*/                   IMAD.IADD R4, R4, 0x1, R5 ;
        /*01d0*/                   IADD3 R2, R2, -0x10, RZ ;
        /*01e0*/                   IMAD.IADD R5, R4, 0x1, R5 ;
        /*01f0*/                   ISETP.GT.AND P1, PT, R2, 0xd, PT ;
        /*0200*/                   IADD3 R4, R4, R5, RZ ;
        /*0210*/                   IMAD.IADD R5, R5, 0x1, R4 ;
        /*0220*/                   IMAD.IADD R4, R4, 0x1, R5 ;
        /*0230*/                   IMAD.IADD R5, R5, 0x1, R4 ;
        /*0240*/                   IADD3 R4, R4, R5, RZ ;
        /*0250*/                   IMAD.IADD R5, R5, 0x1, R4 ;
        /*0260*/                   IMAD.IADD R4, R4, 0x1, R5 ;
        /*0270*/                   IMAD.IADD R5, R5, 0x1, R4 ;
        /*0280*/                   IADD3 R4, R4, R5, RZ ;
        /*0290*/                   IMAD.IADD R5, R5, 0x1, R4 ;
        /*02a0*/                   IMAD.IADD R4, R4, 0x1, R5 ;
        /*02b0*/                   IMAD.IADD R5, R5, 0x1, R4 ;
        /*02c0*/                   IADD3 R4, R4, R5, RZ ;
        /*02d0*/                   IMAD.IADD R5, R5, 0x1, R4 ;
        /*02e0*/               @P1 BRA `(.L_x_4) ;
.L_x_3:
        /*02f0*/                   IADD3 R0, R2, -0x1, RZ ;
        /*0300*/                   ISETP.GT.AND P1, PT, R0, 0x4, PT ;
        /*0310*/              @!P1 BRA `(.L_x_5) ;
        /*0320*/                   IMAD.IADD R4, R5.reuse, 0x1, R4 ;
        /*0330*/                   PLOP3.LUT P0, PT, PT, PT, PT, 0x8, 0x0 ;
        /*0340*/                   IADD3 R2, R2, -0x8, RZ ;
        /*0350*/                   IMAD.IADD R5, R5, 0x1, R4 ;
        /*0360*/                   IADD3 R4, R4, R5, RZ ;
        /*0370*/                   IMAD.IADD R5, R5, 0x1, R4 ;
        /*0380*/                   IMAD.IADD R4, R4, 0x1, R5 ;
        /*0390*/                   IMAD.IADD R5, R5, 0x1, R4 ;
        /*03a0*/                   IADD3 R4, R4, R5, RZ ;
        /*03b0*/                   IMAD.IADD R5, R5, 0x1, R4 ;
.L_x_5:
        /*03c0*/                   ISETP.NE.OR P0, PT, R2, 0x1, P0 ;
        /*03d0*/              @!P0 BRA `(.L_x_1) ;
.L_x_2:
        /*03e0*/                   IADD3 R2, R2, -0x4, RZ ;
        /*03f0*/                   IMAD.IADD R4, R4, 0x1, R5 ;
        /*0400*/                   ISETP.NE.AND P0, PT, R2, 0x1, PT ;
        /*0410*/                   IADD3 R5, R4, R5, RZ ;
        /*0420*/                   IMAD.IADD R4, R4, 0x1, R5 ;
        /*0430*/                   IMAD.IADD R5, R5, 0x1, R4 ;
        /*0440*/               @P0 BRA `(.L_x_2) ;
.L_x_1:
        /*0450*/                   ISETP.NE.AND P0, PT, R3, RZ, PT ;
        /*0460*/              @!P0 BRA `(.L_x_0) ;
.L_x_6:
        /*0470*/                   IADD3 R3, R3, -0x1, RZ ;
        /*0480*/                   MOV R0, R5 ;
        /*0490*/                   IMAD.IADD R5, R4, 0x1, R5 ;
        /*04a0*/                   ISETP.NE.AND P0, PT, R3, RZ, PT ;
        /*04b0*/                   IMAD.MOV.U32 R4, RZ, RZ, R0 ;
        /*04c0*/               @P0 BRA `(.L_x_6) ;
.L_x_0:
        /*04d0*/                   MOV R2, c[0x0][0x160] ;
        /*04e0*/                   IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x164] ;
        /*04f0*/                   STG.E [R2.64], R5 ;
        /*0500*/                   EXIT ;
.L_x_7:
        /*0510*/                   BRA `(.L_x_7);
        /*0520*/                   NOP;
        /*0530*/                   NOP;
        /*0540*/                   NOP;
        /*0550*/                   NOP;
        /*0560*/                   NOP;
        /*0570*/                   NOP;
        /*0580*/                   NOP;
        /*0590*/                   NOP;
        /*05a0*/                   NOP;
        /*05b0*/                   NOP;
        /*05c0*/                   NOP;
        /*05d0*/                   NOP;
        /*05e0*/                   NOP;
        /*05f0*/                   NOP;
.L_x_8:
