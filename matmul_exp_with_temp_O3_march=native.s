	.file	"matmul_experiments.cpp"
	.intel_syntax noprefix
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC3:
	.string	"Time taken(naive-ijk loop with  -O3 optimization flag in compilation flag) for the config  (M=2000,N=2000,K=2000) and with temporary sum_variable(to avoid global ram writes for every accumulation) for accumulations   : "
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC5:
	.string	"secs"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB2138:
	.cfi_startproc
	endbr64
	lea	r10, 8[rsp]
	.cfi_def_cfa 10, 0
	and	rsp, -32
	mov	edi, 16000000
	push	QWORD PTR -8[r10]
	push	rbp
	mov	rbp, rsp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	push	r15
	push	r14
	push	r13
	push	r12
	push	r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	push	rbx
	sub	rsp, 32
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	call	_Znam@PLT
	mov	edi, 16000000
	mov	r13, rax
	call	_Znam@PLT
	mov	edi, 16000000
	mov	r14, rax
	call	_Znam@PLT
	vmovaps	ymm0, YMMWORD PTR .LC1[rip]
	lea	rdx, 16000000[r13]
	mov	r12, rax
	mov	rax, r13
.L2:
	vmovups	YMMWORD PTR [rax], ymm0
	add	rax, 32
	cmp	rdx, rax
	jne	.L2
	mov	rax, r14
	lea	r15, 16000000[r14]
.L3:
	vmovups	YMMWORD PTR [rax], ymm0
	add	rax, 32
	cmp	rax, r15
	jne	.L3
	mov	ebx, 3
.L9:
	mov	edx, 16000000
	xor	esi, esi
	mov	rdi, r12
	vzeroupper
	call	memset@PLT
	mov	r8, r13
	mov	rsi, r12
	xor	edi, edi
.L4:
	mov	rcx, r15
	lea	r9, 8000[rsi]
.L8:
	lea	rax, -16000000[rcx]
	mov	rdx, r8
	vxorps	xmm0, xmm0, xmm0
	.p2align 4,,10
	.p2align 3
.L5:
	vbroadcastss	ymm1, DWORD PTR [rdx]
	add	rax, 8000
	add	rdx, 4
	vfmadd231ps	ymm0, ymm1, YMMWORD PTR -8000[rax]
	cmp	rax, rcx
	jne	.L5
	vmovups	YMMWORD PTR [rsi], ymm0
	add	rsi, 32
	lea	rcx, 32[rax]
	cmp	rsi, r9
	jne	.L8
	add	edi, 2000
	add	r8, 8000
	cmp	edi, 4000000
	jne	.L4
	sub	ebx, 1
	jne	.L9
	vxorpd	xmm2, xmm2, xmm2
	mov	ebx, 10
	vmovsd	QWORD PTR -56[rbp], xmm2
	vzeroupper
.L15:
	xor	esi, esi
	mov	rdi, r12
	mov	edx, 16000000
	call	memset@PLT
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	mov	rsi, r12
	xor	edi, edi
	mov	r8, rax
.L10:
	imul	r9, rdi, 8000
	mov	rcx, r15
	lea	r10, 8000[rsi]
	add	r9, r13
.L14:
	lea	rax, -16000000[rcx]
	mov	rdx, r9
	vxorps	xmm0, xmm0, xmm0
	.p2align 4,,10
	.p2align 3
.L11:
	vbroadcastss	ymm1, DWORD PTR [rdx]
	add	rax, 8000
	add	rdx, 4
	vfmadd231ps	ymm0, ymm1, YMMWORD PTR -8000[rax]
	cmp	rax, rcx
	jne	.L11
	vmovups	YMMWORD PTR [rsi], ymm0
	add	rsi, 32
	lea	rcx, 32[rax]
	cmp	rsi, r10
	jne	.L14
	add	rdi, 1
	cmp	rdi, 2000
	jne	.L10
	mov	QWORD PTR -64[rbp], r8
	vzeroupper
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	mov	r8, QWORD PTR -64[rbp]
	vxorpd	xmm3, xmm3, xmm3
	sub	rax, r8
	vcvtsi2sd	xmm0, xmm3, rax
	vdivsd	xmm0, xmm0, QWORD PTR .LC2[rip]
	vaddsd	xmm4, xmm0, QWORD PTR -56[rbp]
	vmovsd	QWORD PTR -56[rbp], xmm4
	sub	ebx, 1
	jne	.L15
	lea	r15, _ZSt4cout[rip]
	mov	edx, 219
	lea	rsi, .LC3[rip]
	mov	rdi, r15
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	mov	rdi, r15
	vmovsd	xmm6, QWORD PTR -56[rbp]
	vdivsd	xmm0, xmm6, QWORD PTR .LC4[rip]
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	mov	edx, 4
	lea	rsi, .LC5[rip]
	mov	r15, rax
	mov	rdi, rax
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	mov	rax, QWORD PTR [r15]
	mov	rax, QWORD PTR -24[rax]
	mov	rbx, QWORD PTR 240[r15+rax]
	test	rbx, rbx
	je	.L27
	cmp	BYTE PTR 56[rbx], 0
	je	.L17
	movzx	eax, BYTE PTR 67[rbx]
.L18:
	movsx	esi, al
	mov	rdi, r15
	call	_ZNSo3putEc@PLT
	mov	rdi, rax
	call	_ZNSo5flushEv@PLT
	mov	rdi, r13
	call	_ZdaPv@PLT
	mov	rdi, r14
	call	_ZdaPv@PLT
	mov	rdi, r12
	call	_ZdaPv@PLT
	add	rsp, 32
	xor	eax, eax
	pop	rbx
	pop	r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	lea	rsp, -8[r10]
	.cfi_def_cfa 7, 8
	ret
.L17:
	.cfi_restore_state
	mov	rdi, rbx
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	mov	rax, QWORD PTR [rbx]
	mov	esi, 10
	mov	rdi, rbx
	call	[QWORD PTR 48[rax]]
	jmp	.L18
.L27:
	call	_ZSt16__throw_bad_castv@PLT
	.cfi_endproc
.LFE2138:
	.size	main, .-main
	.p2align 4
	.type	_GLOBAL__sub_I_main, @function
_GLOBAL__sub_I_main:
.LFB2672:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	lea	rbp, _ZStL8__ioinit[rip]
	mov	rdi, rbp
	call	_ZNSt8ios_base4InitC1Ev@PLT
	mov	rdi, QWORD PTR _ZNSt8ios_base4InitD1Ev@GOTPCREL[rip]
	mov	rsi, rbp
	pop	rbp
	.cfi_def_cfa_offset 8
	lea	rdx, __dso_handle[rip]
	jmp	__cxa_atexit@PLT
	.cfi_endproc
.LFE2672:
	.size	_GLOBAL__sub_I_main, .-_GLOBAL__sub_I_main
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_main
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC1:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC2:
	.long	0
	.long	1104006501
	.align 8
.LC4:
	.long	0
	.long	1076101120
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
