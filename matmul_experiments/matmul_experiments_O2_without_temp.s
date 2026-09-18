	.file	"matmul_experiments.cpp"
	.intel_syntax noprefix
	.text
	.section	.text._ZNKSt5ctypeIcE8do_widenEc,"axG",@progbits,_ZNKSt5ctypeIcE8do_widenEc,comdat
	.align 2
	.p2align 4
	.weak	_ZNKSt5ctypeIcE8do_widenEc
	.type	_ZNKSt5ctypeIcE8do_widenEc, @function
_ZNKSt5ctypeIcE8do_widenEc:
.LFB1891:
	.cfi_startproc
	endbr64
	mov	eax, esi
	ret
	.cfi_endproc
.LFE1891:
	.size	_ZNKSt5ctypeIcE8do_widenEc, .-_ZNKSt5ctypeIcE8do_widenEc
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC4:
	.string	"Time taken(naive-ijk loop with  -O2 optimization flag in compilation flag) for the config  (M=2000,N=2000,K=2000) and without temporary sum_variable(to avoid global ram writes for every accumulation) for accumulations   : "
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC6:
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
	movss	xmm0, DWORD PTR .LC2[rip]
	lea	rdx, 16000000[r12]
	mov	rbp, rax
	mov	rax, r12
.L4:
	movss	DWORD PTR [rax], xmm0
	add	rax, 4
	cmp	rax, rdx
	jne	.L4
	mov	rax, r13
	lea	r14, 16000000[r13]
.L5:
	movss	DWORD PTR [rax], xmm0
	add	rax, 4
	cmp	rax, r14
	jne	.L5
	mov	r15d, 3
	lea	rbx, 16000000[rbp]
.L11:
	mov	rdi, rbp
	mov	edx, 16000000
	xor	esi, esi
	call	memset@PLT
	mov	r8, r12
	mov	rdi, rbp
	pxor	xmm1, xmm1
.L6:
	mov	rcx, r14
	xor	esi, esi
.L10:
	lea	rax, -16000000[rcx]
	mov	rdx, r8
	.p2align 4,,10
	.p2align 3
.L7:
	movss	xmm0, DWORD PTR [rdx]
	mulss	xmm0, DWORD PTR [rax]
	add	rax, 8000
	add	rdx, 4
	addss	xmm1, xmm0
	cmp	rcx, rax
	jne	.L7
	movss	DWORD PTR [rdi+rsi*4], xmm1
	add	rsi, 1
	add	rcx, 4
	cmp	rsi, 2000
	je	.L8
	movss	xmm1, DWORD PTR [rdi+rsi*4]
	jmp	.L10
.L8:
	add	rdi, 8000
	add	r8, 8000
	cmp	rdi, rbx
	je	.L9
	movss	xmm1, DWORD PTR [rdi]
	jmp	.L6
.L9:
	sub	r15d, 1
	jne	.L11
	pxor	xmm2, xmm2
	mov	ebx, 10
	movsd	QWORD PTR 8[rsp], xmm2
.L17:
	mov	edx, 16000000
	xor	esi, esi
	mov	rdi, rbp
	call	memset@PLT
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	xor	r10d, r10d
	xor	r9d, r9d
	pxor	xmm1, xmm1
	mov	r15, rax
.L12:
	mov	rcx, r14
	lea	rdi, 0[rbp+r9]
	lea	r8, [r12+r9]
	xor	esi, esi
.L16:
	mov	rdx, r8
	lea	rax, -16000000[rcx]
	.p2align 4,,10
	.p2align 3
.L13:
	movss	xmm0, DWORD PTR [rdx]
	mulss	xmm0, DWORD PTR [rax]
	add	rax, 8000
	add	rdx, 4
	addss	xmm1, xmm0
	cmp	rcx, rax
	jne	.L13
	movss	DWORD PTR [rdi+rsi*4], xmm1
	add	rsi, 1
	add	rcx, 4
	cmp	rsi, 2000
	je	.L14
	movss	xmm1, DWORD PTR [rdi+rsi*4]
	jmp	.L16
.L14:
	add	r10d, 2000
	add	r9, 8000
	cmp	r10d, 4000000
	je	.L15
	movss	xmm1, DWORD PTR 0[rbp+r9]
	jmp	.L12
.L15:
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	pxor	xmm0, xmm0
	sub	rax, r15
	cvtsi2sd	xmm0, rax
	divsd	xmm0, QWORD PTR .LC3[rip]
	addsd	xmm0, QWORD PTR 8[rsp]
	movsd	QWORD PTR 8[rsp], xmm0
	sub	ebx, 1
	jne	.L17
	lea	rsi, .LC4[rip]
	lea	rdi, _ZSt4cout[rip]
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	movsd	xmm0, QWORD PTR 8[rsp]
	divsd	xmm0, QWORD PTR .LC5[rip]
	mov	rdi, rax
	call	_ZNSo9_M_insertIdEERSoT_@PLT
	lea	rsi, .LC6[rip]
	mov	rdi, rax
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	mov	r15, rax
	mov	rax, QWORD PTR [rax]
	mov	rax, QWORD PTR -24[rax]
	mov	r14, QWORD PTR 240[r15+rax]
	test	r14, r14
	je	.L29
	cmp	BYTE PTR 56[r14], 0
	je	.L19
	movzx	eax, BYTE PTR 67[r14]
.L20:
	movsx	esi, al
	mov	rdi, r15
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
.L19:
	.cfi_restore_state
	mov	rdi, r14
	call	_ZNKSt5ctypeIcE13_M_widen_initEv@PLT
	mov	rax, QWORD PTR [r14]
	lea	rcx, _ZNKSt5ctypeIcE8do_widenEc[rip]
	mov	rdx, QWORD PTR 48[rax]
	mov	eax, 10
	cmp	rdx, rcx
	je	.L20
	mov	esi, 10
	mov	rdi, r14
	call	rdx
	jmp	.L20
.L29:
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
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC2:
	.long	1065353216
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC3:
	.long	0
	.long	1104006501
	.align 8
.LC5:
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
