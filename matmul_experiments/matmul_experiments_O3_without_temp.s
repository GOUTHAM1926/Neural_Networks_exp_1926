	.file	"matmul_experiments.cpp"
	.intel_syntax noprefix
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC3:
	.string	"Time taken(naive-ijk loop with  -O3 optimization flag in compilation flag) for the config  (M=2000,N=2000,K=2000) and without temporary sum_variable(to avoid global ram writes for every accumulation) for accumulations   : "
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
	push	r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	mov	edi, 16000000
	push	r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	push	r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	push	r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	push	rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	push	rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	sub	rsp, 24
	.cfi_def_cfa_offset 80
	call	_Znam@PLT
	mov	edi, 16000000
	mov	r12, rax
	call	_Znam@PLT
	mov	edi, 16000000
	mov	r13, rax
	call	_Znam@PLT
	movaps	xmm0, XMMWORD PTR .LC1[rip]
	lea	rdx, 16000000[r12]
	mov	rbp, rax
	mov	rax, r12
.L2:
	movups	XMMWORD PTR [rax], xmm0
	add	rax, 16
	cmp	rax, rdx
	jne	.L2
	mov	rax, r13
	lea	r14, 16000000[r13]
.L3:
	movups	XMMWORD PTR [rax], xmm0
	add	rax, 16
	cmp	rax, r14
	jne	.L3
	mov	ebx, 3
.L9:
	mov	rdi, rbp
	mov	edx, 16000000
	xor	esi, esi
	call	memset@PLT
	mov	r9, r12
	mov	rdi, rbp
	xor	r8d, r8d
.L4:
	mov	rcx, r14
	xor	esi, esi
.L8:
	movups	xmm1, XMMWORD PTR [rdi+rsi]
	lea	rax, -16000000[rcx]
	mov	rdx, r9
	.p2align 4,,10
	.p2align 3
.L5:
	movss	xmm0, DWORD PTR [rdx]
	movups	xmm3, XMMWORD PTR [rax]
	add	rax, 8000
	add	rdx, 4
	shufps	xmm0, xmm0, 0
	mulps	xmm0, xmm3
	addps	xmm1, xmm0
	cmp	rax, rcx
	jne	.L5
	movups	XMMWORD PTR [rdi+rsi], xmm1
	add	rsi, 16
	lea	rcx, 16[rax]
	cmp	rsi, 8000
	jne	.L8
	add	r8d, 2000
	add	rdi, 8000
	add	r9, 8000
	cmp	r8d, 4000000
	jne	.L4
	sub	ebx, 1
	jne	.L9
	pxor	xmm4, xmm4
	mov	ebx, 10
	movsd	QWORD PTR 8[rsp], xmm4
.L15:
	mov	rdi, rbp
	mov	edx, 16000000
	xor	esi, esi
	call	memset@PLT
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	mov	rdi, rbp
	xor	r8d, r8d
	mov	r15, rax
.L10:
	imul	r9, r8, 8000
	mov	rcx, r14
	xor	esi, esi
	add	r9, r12
.L14:
	movups	xmm1, XMMWORD PTR [rdi+rsi]
	lea	rax, -16000000[rcx]
	mov	rdx, r9
	.p2align 4,,10
	.p2align 3
.L11:
	movss	xmm0, DWORD PTR [rdx]
	movups	xmm2, XMMWORD PTR [rax]
	add	rax, 8000
	add	rdx, 4
	shufps	xmm0, xmm0, 0
	mulps	xmm0, xmm2
	addps	xmm1, xmm0
	cmp	rax, rcx
	jne	.L11
	movups	XMMWORD PTR [rdi+rsi], xmm1
	add	rsi, 16
	lea	rcx, 16[rax]
	cmp	rsi, 8000
	jne	.L14
	add	r8, 1
	add	rdi, 8000
	cmp	r8, 2000
	jne	.L10
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	pxor	xmm0, xmm0
	sub	rax, r15
	cvtsi2sd	xmm0, rax
	divsd	xmm0, QWORD PTR .LC2[rip]
	addsd	xmm0, QWORD PTR 8[rsp]
	movsd	QWORD PTR 8[rsp], xmm0
	sub	ebx, 1
	jne	.L15
	lea	r14, _ZSt4cout[rip]
	mov	edx, 222
	lea	rsi, .LC3[rip]
	mov	rdi, r14
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	mov	rdi, r14
	movsd	xmm0, QWORD PTR 8[rsp]
	divsd	xmm0, QWORD PTR .LC4[rip]
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	mov	edx, 4
	lea	rsi, .LC5[rip]
	mov	r14, rax
	mov	rdi, rax
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	mov	rax, QWORD PTR [r14]
	mov	rax, QWORD PTR -24[rax]
	mov	r15, QWORD PTR 240[r14+rax]
	test	r15, r15
	je	.L27
	cmp	BYTE PTR 56[r15], 0
	je	.L17
	movzx	eax, BYTE PTR 67[r15]
.L18:
	movsx	esi, al
	mov	rdi, r14
	call	_ZNSo3putEc@PLT
	mov	rdi, rax
	call	_ZNSo5flushEv@PLT
	mov	rdi, r12
	call	_ZdaPv@PLT
	mov	rdi, r13
	call	_ZdaPv@PLT
	mov	rdi, rbp
	call	_ZdaPv@PLT
	add	rsp, 24
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	xor	eax, eax
	pop	rbx
	.cfi_def_cfa_offset 48
	pop	rbp
	.cfi_def_cfa_offset 40
	pop	r12
	.cfi_def_cfa_offset 32
	pop	r13
	.cfi_def_cfa_offset 24
	pop	r14
	.cfi_def_cfa_offset 16
	pop	r15
	.cfi_def_cfa_offset 8
	ret
.L17:
	.cfi_restore_state
	mov	rdi, r15
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	mov	rax, QWORD PTR [r15]
	mov	esi, 10
	mov	rdi, r15
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
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC1:
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
