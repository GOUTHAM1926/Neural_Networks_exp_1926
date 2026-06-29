	.file	"matmul_experiments.cpp"
	.intel_syntax noprefix
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC3:
	.string	"Time taken(naive-ijk loop with  -O1 optimization flag in compilation flag) for the config  (M=2000,N=2000,K=2000) and with  no temporary sum_variable(to avoid global ram writes for every accumulation) for accumulations   : "
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC5:
	.string	"secs"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2138:
	.cfi_startproc
	endbr64
	push	r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
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
	mov	edi, 16000000
	call	_Znam@PLT
	mov	rbp, rax
	mov	edi, 16000000
	call	_Znam@PLT
	mov	r12, rax
	mov	edi, 16000000
	call	_Znam@PLT
	mov	rbx, rax
	lea	rdx, 16000000[rbp]
	mov	rax, rbp
	movss	xmm0, DWORD PTR .LC1[rip]
.L2:
	movss	DWORD PTR [rax], xmm0
	add	rax, 4
	cmp	rax, rdx
	jne	.L2
	mov	rax, r12
	lea	r14, 16000000[r12]
	movss	xmm0, DWORD PTR .LC1[rip]
.L3:
	movss	DWORD PTR [rax], xmm0
	add	rax, 4
	cmp	rax, r14
	jne	.L3
	mov	r13d, 3
	lea	r15, 16000000[rbx]
.L9:
	mov	edx, 16000000
	mov	esi, 0
	mov	rdi, rbx
	call	memset@PLT
	mov	r8, rbp
	mov	rdi, rbx
.L4:
	mov	rcx, r14
	mov	esi, 0
.L8:
	mov	r9, rdi
	movss	xmm1, DWORD PTR [rdi+rsi*4]
	lea	rax, -16000000[rcx]
	mov	rdx, r8
.L5:
	movss	xmm0, DWORD PTR [rdx]
	mulss	xmm0, DWORD PTR [rax]
	addss	xmm1, xmm0
	add	rdx, 4
	add	rax, 8000
	cmp	rax, rcx
	jne	.L5
	movss	DWORD PTR [r9+rsi*4], xmm1
	add	rsi, 1
	add	rcx, 4
	cmp	rsi, 2000
	jne	.L8
	add	rdi, 8000
	add	r8, 8000
	cmp	rdi, r15
	jne	.L4
	sub	r13d, 1
	jne	.L9
	mov	r13d, 10
	mov	r15, QWORD PTR .LC0[rip]
	jmp	.L15
.L23:
	movss	DWORD PTR [rdi+rsi*4], xmm1
	add	rsi, 1
	add	rcx, 4
	cmp	rsi, 2000
	je	.L12
.L14:
	movss	xmm1, DWORD PTR [rdi+rsi*4]
	lea	rax, -16000000[rcx]
	mov	rdx, r8
.L11:
	movss	xmm0, DWORD PTR [rdx]
	mulss	xmm0, DWORD PTR [rax]
	addss	xmm1, xmm0
	add	rdx, 4
	add	rax, 8000
	cmp	rax, rcx
	jne	.L11
	jmp	.L23
.L12:
	add	r9, 1
	cmp	r9, 2000
	je	.L13
.L10:
	imul	r8, r9, 8000
	lea	rdi, [rbx+r8]
	add	r8, rbp
	mov	rcx, r14
	mov	esi, 0
	jmp	.L14
.L13:
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	sub	rax, QWORD PTR 8[rsp]
	pxor	xmm0, xmm0
	cvtsi2sd	xmm0, rax
	divsd	xmm0, QWORD PTR .LC2[rip]
	movq	xmm2, r15
	addsd	xmm2, xmm0
	movq	r15, xmm2
	sub	r13d, 1
	je	.L24
.L15:
	mov	edx, 16000000
	mov	esi, 0
	mov	rdi, rbx
	call	memset@PLT
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	mov	QWORD PTR 8[rsp], rax
	mov	r9d, 0
	jmp	.L10
.L24:
	lea	rsi, .LC3[rip]
	lea	rdi, _ZSt4cout[rip]
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	mov	rdi, rax
	movq	xmm0, r15
	divsd	xmm0, QWORD PTR .LC4[rip]
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	mov	rdi, rax
	lea	rsi, .LC5[rip]
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	mov	rdi, rax
	call	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@PLT
	mov	rdi, rbp
	call	_ZdaPv@PLT
	mov	rdi, r12
	call	_ZdaPv@PLT
	mov	rdi, rbx
	call	_ZdaPv@PLT
	mov	eax, 0
	add	rsp, 24
	.cfi_def_cfa_offset 56
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
	.cfi_endproc
.LFE2138:
	.size	main, .-main
	.type	_GLOBAL__sub_I_main, @function
_GLOBAL__sub_I_main:
.LFB2672:
	.cfi_startproc
	endbr64
	push	rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	lea	rbx, _ZStL8__ioinit[rip]
	mov	rdi, rbx
	call	_ZNSt8ios_base4InitC1Ev@PLT
	lea	rdx, __dso_handle[rip]
	mov	rsi, rbx
	mov	rdi, QWORD PTR _ZNSt8ios_base4InitD1Ev@GOTPCREL[rip]
	call	__cxa_atexit@PLT
	pop	rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE2672:
	.size	_GLOBAL__sub_I_main, .-_GLOBAL__sub_I_main
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_main
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	0
	.long	0
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC1:
	.long	1065353216
	.section	.rodata.cst8
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
