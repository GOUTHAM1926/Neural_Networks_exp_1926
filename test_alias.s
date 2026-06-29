	.file	"test_alias.cpp"
	.intel_syntax noprefix
	.text
	.globl	_Z11matmul_funcPfS_S_iii
	.type	_Z11matmul_funcPfS_S_iii, @function
_Z11matmul_funcPfS_S_iii:
.LFB0:
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
	mov	QWORD PTR -24[rsp], rsi
	mov	DWORD PTR -28[rsp], ecx
	test	ecx, ecx
	jle	.L1
	mov	rbx, rdi
	mov	r12, rdx
	mov	r10d, r8d
	mov	r11d, r9d
	movsx	rdi, r8d
	sal	rdi, 2
	mov	ebp, 0
	mov	r15d, 0
	mov	r14d, 0
	lea	eax, -1[r9]
	mov	QWORD PTR -16[rsp], rax
	lea	rax, 4[rbx]
	mov	QWORD PTR -8[rsp], rax
	jmp	.L3
.L6:
	lea	eax, [r8+rbp]
	cdqe
	lea	rcx, [r12+rax*4]
	mov	rdx, r9
	mov	rax, r13
.L4:
	movss	xmm0, DWORD PTR [rax]
	mulss	xmm0, DWORD PTR [rdx]
	addss	xmm0, DWORD PTR [rcx]
	movss	DWORD PTR [rcx], xmm0
	add	rax, 4
	add	rdx, rdi
	cmp	rax, rsi
	jne	.L4
.L7:
	add	r8d, 1
	add	r9, 4
	cmp	r10d, r8d
	je	.L5
.L8:
	test	r11d, r11d
	jg	.L6
	jmp	.L7
.L5:
	add	r14d, 1
	add	r15d, r11d
	add	ebp, r10d
	cmp	DWORD PTR -28[rsp], r14d
	je	.L1
.L3:
	test	r10d, r10d
	jle	.L5
	mov	r9, QWORD PTR -24[rsp]
	movsx	rax, r15d
	lea	r13, [rbx+rax*4]
	add	rax, QWORD PTR -16[rsp]
	mov	rsi, QWORD PTR -8[rsp]
	lea	rsi, [rsi+rax*4]
	mov	r8d, 0
	jmp	.L8
.L1:
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
.LFE0:
	.size	_Z11matmul_funcPfS_S_iii, .-_Z11matmul_funcPfS_S_iii
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
