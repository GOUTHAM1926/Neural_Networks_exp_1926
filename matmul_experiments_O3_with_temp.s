	.file	"matmul_experiments.cpp"
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
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movl	$16000000, %edi
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
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	call	_Znam@PLT
	movl	$16000000, %edi
	movq	%rax, %r12
	call	_Znam@PLT
	movl	$16000000, %edi
	movq	%rax, %r13
	call	_Znam@PLT
	movaps	.LC1(%rip), %xmm0
	leaq	16000000(%r12), %rdx
	movq	%rax, %rbp
	movq	%r12, %rax
.L2:
	movups	%xmm0, (%rax)
	addq	$16, %rax
	cmpq	%rax, %rdx
	jne	.L2
	movq	%r13, %rax
	leaq	16000000(%r13), %r14
.L3:
	movups	%xmm0, (%rax)
	addq	$16, %rax
	cmpq	%r14, %rax
	jne	.L3
	movl	$3, %ebx
.L9:
	xorl	%esi, %esi
	movq	%rbp, %rdi
	movl	$16000000, %edx
	call	memset@PLT
	movq	%r12, %r8
	movq	%rbp, %rsi
	xorl	%edi, %edi
.L4:
	movq	%r14, %rcx
	leaq	8000(%rsi), %r9
.L8:
	leaq	-16000000(%rcx), %rax
	movq	%r8, %rdx
	pxor	%xmm1, %xmm1
	.p2align 4,,10
	.p2align 3
.L5:
	movss	(%rdx), %xmm0
	movups	(%rax), %xmm3
	addq	$8000, %rax
	addq	$4, %rdx
	shufps	$0, %xmm0, %xmm0
	mulps	%xmm3, %xmm0
	addps	%xmm0, %xmm1
	cmpq	%rcx, %rax
	jne	.L5
	movups	%xmm1, (%rsi)
	addq	$16, %rsi
	leaq	16(%rax), %rcx
	cmpq	%r9, %rsi
	jne	.L8
	addl	$2000, %edi
	addq	$8000, %r8
	cmpl	$4000000, %edi
	jne	.L4
	subl	$1, %ebx
	jne	.L9
	pxor	%xmm4, %xmm4
	movl	$10, %ebx
	movsd	%xmm4, 8(%rsp)
.L15:
	xorl	%esi, %esi
	movq	%rbp, %rdi
	movl	$16000000, %edx
	call	memset@PLT
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	movq	%rbp, %rsi
	xorl	%edi, %edi
	movq	%rax, %r15
.L10:
	imulq	$8000, %rdi, %r8
	movq	%r14, %rcx
	leaq	8000(%rsi), %r9
	addq	%r12, %r8
.L14:
	leaq	-16000000(%rcx), %rax
	movq	%r8, %rdx
	pxor	%xmm1, %xmm1
	.p2align 4,,10
	.p2align 3
.L11:
	movss	(%rdx), %xmm0
	movups	(%rax), %xmm2
	addq	$8000, %rax
	addq	$4, %rdx
	shufps	$0, %xmm0, %xmm0
	mulps	%xmm2, %xmm0
	addps	%xmm0, %xmm1
	cmpq	%rcx, %rax
	jne	.L11
	movups	%xmm1, (%rsi)
	addq	$16, %rsi
	leaq	16(%rax), %rcx
	cmpq	%r9, %rsi
	jne	.L14
	addq	$1, %rdi
	cmpq	$2000, %rdi
	jne	.L10
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	pxor	%xmm0, %xmm0
	subq	%r15, %rax
	cvtsi2sdq	%rax, %xmm0
	divsd	.LC2(%rip), %xmm0
	addsd	8(%rsp), %xmm0
	movsd	%xmm0, 8(%rsp)
	subl	$1, %ebx
	jne	.L15
	leaq	_ZSt4cout(%rip), %r14
	movl	$219, %edx
	leaq	.LC3(%rip), %rsi
	movq	%r14, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	%r14, %rdi
	movsd	8(%rsp), %xmm0
	divsd	.LC4(%rip), %xmm0
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	movl	$4, %edx
	leaq	.LC5(%rip), %rsi
	movq	%rax, %r14
	movq	%rax, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movq	(%r14), %rax
	movq	-24(%rax), %rax
	movq	240(%r14,%rax), %r15
	testq	%r15, %r15
	je	.L27
	cmpb	$0, 56(%r15)
	je	.L17
	movzbl	67(%r15), %eax
.L18:
	movsbl	%al, %esi
	movq	%r14, %rdi
	call	_ZNSo3putEc@PLT
	movq	%rax, %rdi
	call	_ZNSo5flushEv@PLT
	movq	%r12, %rdi
	call	_ZdaPv@PLT
	movq	%r13, %rdi
	call	_ZdaPv@PLT
	movq	%rbp, %rdi
	call	_ZdaPv@PLT
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	xorl	%eax, %eax
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
.L17:
	.cfi_restore_state
	movq	%r15, %rdi
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	movq	(%r15), %rax
	movl	$10, %esi
	movq	%r15, %rdi
	call	*48(%rax)
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
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	leaq	_ZStL8__ioinit(%rip), %rbp
	movq	%rbp, %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	movq	%rbp, %rsi
	popq	%rbp
	.cfi_def_cfa_offset 8
	leaq	__dso_handle(%rip), %rdx
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
