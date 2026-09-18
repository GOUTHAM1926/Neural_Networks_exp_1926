	.file	"test_alias.cpp"
	.text
	.globl	_Z11matmul_funcPfS_S_iii
	.type	_Z11matmul_funcPfS_S_iii, @function
_Z11matmul_funcPfS_S_iii:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rsi, -24(%rsp)
	movl	%ecx, -28(%rsp)
	testl	%ecx, %ecx
	jle	.L1
	movq	%rdi, %rbx
	movq	%rdx, %r12
	movl	%r8d, %r10d
	movl	%r9d, %r11d
	movslq	%r8d, %rdi
	salq	$2, %rdi
	movl	$0, %ebp
	movl	$0, %r15d
	movl	$0, %r14d
	leal	-1(%r9), %eax
	movq	%rax, -16(%rsp)
	leaq	4(%rbx), %rax
	movq	%rax, -8(%rsp)
	jmp	.L3
.L6:
	leal	(%r8,%rbp), %eax
	cltq
	leaq	(%r12,%rax,4), %rcx
	movq	%r9, %rdx
	movq	%r13, %rax
.L4:
	movss	(%rax), %xmm0
	mulss	(%rdx), %xmm0
	addss	(%rcx), %xmm0
	movss	%xmm0, (%rcx)
	addq	$4, %rax
	addq	%rdi, %rdx
	cmpq	%rsi, %rax
	jne	.L4
.L7:
	addl	$1, %r8d
	addq	$4, %r9
	cmpl	%r8d, %r10d
	je	.L5
.L8:
	testl	%r11d, %r11d
	jg	.L6
	jmp	.L7
.L5:
	addl	$1, %r14d
	addl	%r11d, %r15d
	addl	%r10d, %ebp
	cmpl	%r14d, -28(%rsp)
	je	.L1
.L3:
	testl	%r10d, %r10d
	jle	.L5
	movq	-24(%rsp), %r9
	movslq	%r15d, %rax
	leaq	(%rbx,%rax,4), %r13
	addq	-16(%rsp), %rax
	movq	-8(%rsp), %rsi
	leaq	(%rsi,%rax,4), %rsi
	movl	$0, %r8d
	jmp	.L8
.L1:
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
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
