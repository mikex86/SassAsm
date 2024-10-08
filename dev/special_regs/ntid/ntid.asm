	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM89)"
	.elftype	@"ET_EXEC"


//--------------------- .debug_frame              --------------------------
	.section	.debug_frame,"",@progbits
.debug_frame:
        /*0000*/ 	.byte	0xff, 0xff, 0xff, 0xff, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff
        /*0010*/ 	.byte	0xff, 0xff, 0xff, 0xff, 0x03, 0x00, 0x04, 0x7c, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x0c, 0x81, 0x80
        /*0020*/ 	.byte	0x80, 0x28, 0x00, 0x08, 0xff, 0x81, 0x80, 0x28, 0x08, 0x81, 0x80, 0x80, 0x28, 0x00, 0x00, 0x00
        /*0030*/ 	.byte	0xff, 0xff, 0xff, 0xff, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        /*0040*/ 	.byte	0x00, 0x00, 0x00, 0x00
        /*0044*/ 	.dword	get_ntid_kernel
        /*004c*/ 	.byte	0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x00, 0x00, 0x00, 0x04, 0x14, 0x00
        /*005c*/ 	.byte	0x00, 0x00, 0x0c, 0x81, 0x80, 0x80, 0x28, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        /*006c*/ 	.byte	0x00, 0x00, 0x00, 0x00


//--------------------- .nv.info                  --------------------------
	.section	.nv.info,"",@"SHT_CUDA_INFO"
	.align	4


	//----- nvinfo : EIATTR_REGCOUNT
	.align		4
        /*0000*/ 	.byte	0x04, 0x2f
        /*0002*/ 	.short	(.L_1 - .L_0)
	.align		4
.L_0:
        /*0004*/ 	.word	index@(get_ntid_kernel)
        /*0008*/ 	.word	0x00000004


	//----- nvinfo : EIATTR_FRAME_SIZE
	.align		4
.L_1:
        /*000c*/ 	.byte	0x04, 0x11
        /*000e*/ 	.short	(.L_3 - .L_2)
	.align		4
.L_2:
        /*0010*/ 	.word	index@(get_ntid_kernel)
        /*0014*/ 	.word	0x00000000


	//----- nvinfo : EIATTR_MIN_STACK_SIZE
	.align		4
.L_3:
        /*0018*/ 	.byte	0x04, 0x12
        /*001a*/ 	.short	(.L_5 - .L_4)
	.align		4
.L_4:
        /*001c*/ 	.word	index@(get_ntid_kernel)
        /*0020*/ 	.word	0x00000000
.L_5:


//--------------------- .nv.info.get_ntid_kernel  --------------------------
	.section	.nv.info.get_ntid_kernel,"",@"SHT_CUDA_INFO"
	.sectionflags	@""
	.align	4


	//----- nvinfo : EIATTR_CUDA_API_VERSION
	.align		4
        /*0000*/ 	.byte	0x04, 0x37
        /*0002*/ 	.short	(.L_7 - .L_6)
.L_6:
        /*0004*/ 	.word	0x0000007e


	//----- nvinfo : EIATTR_MAXREG_COUNT
	.align		4
.L_7:
        /*0008*/ 	.byte	0x03, 0x1b
        /*000a*/ 	.short	0x00ff


	//----- nvinfo : EIATTR_EXIT_INSTR_OFFSETS
	.align		4
        /*000c*/ 	.byte	0x04, 0x1c
        /*000e*/ 	.short	(.L_9 - .L_8)


	//   ....[0]....
.L_8:
        /*0010*/ 	.word	0x00000050
.L_9:


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


//--------------------- .nv.rel.action            --------------------------
	.section	.nv.rel.action,"",@"SHT_CUDA_RELOCINFO"
	.align	8
	.sectionentsize	8
        /*0000*/ 	.byte	0x73, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x25, 0x00, 0x05, 0x36


//--------------------- .nv.constant0.get_ntid_kernel --------------------------
	.section	.nv.constant0.get_ntid_kernel,"a",@progbits
	.sectionflags	@""
	.align	4
.nv.constant0.get_ntid_kernel:
	.zero		352


//--------------------- .text.get_ntid_kernel     --------------------------
	.section	.text.get_ntid_kernel,"ax",@progbits
	.sectioninfo	@"SHI_REGISTERS=4"
	.align	128
        .global         get_ntid_kernel
        .type           get_ntid_kernel,@function
        .size           get_ntid_kernel,(.L_x_1 - get_ntid_kernel)
        .other          get_ntid_kernel,@"STO_CUDA_ENTRY STV_DEFAULT"
get_ntid_kernel:
.text.get_ntid_kernel:
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   ULDC.64 UR4, c[0x0][0x0] ;
        /*0020*/                   IADD3 R0, RZ, -c[0x0][0x8], RZ ;
        /*0030*/                   UIADD3 UR4, UR4, UR5, URZ ;
        /*0040*/                   ISETP.NE.AND P0, PT, R0, UR4, PT ;
        /*0050*/               @P0 EXIT ;
        /*0060*/                   BPT.TRAP 0x1 ;
.L_x_0:
        /*0070*/                   BRA `(.L_x_0);
        /*0080*/                   NOP;
        /*0090*/                   NOP;
        /*00a0*/                   NOP;
        /*00b0*/                   NOP;
        /*00c0*/                   NOP;
        /*00d0*/                   NOP;
        /*00e0*/                   NOP;
        /*00f0*/                   NOP;
.L_x_1:
