	.file	"matmul_experiments.cpp"
	.intel_syntax noprefix
	.text
	.section	.text._ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv,"axG",@progbits,_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv,comdat
	.align 2
	.weak	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv
	.type	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv, @function
_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv:
.LFB271:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	mov	QWORD PTR -8[rbp], rdi
	mov	rax, QWORD PTR -8[rbp]
	mov	rax, QWORD PTR [rax]
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE271:
	.size	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv, .-_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv
	.section	.text._ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_,"axG",@progbits,_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_,comdat
	.align 2
	.weak	_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_
	.type	_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_, @function
_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_:
.LFB276:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	mov	QWORD PTR -8[rbp], rdi
	mov	QWORD PTR -16[rbp], rsi
	mov	rax, QWORD PTR -16[rbp]
	mov	rdx, QWORD PTR [rax]
	mov	rax, QWORD PTR -8[rbp]
	mov	QWORD PTR [rax], rdx
	nop
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE276:
	.size	_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_, .-_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.section	.text._ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC1IlS1_ILl1ELl1000000000EEvEERKNS0_IT_T0_EE,"axG",@progbits,_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC1IlS1_ILl1ELl1000000000EEvEERKNS0_IT_T0_EE,comdat
	.align 2
	.weak	_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC1IlS1_ILl1ELl1000000000EEvEERKNS0_IT_T0_EE
	.type	_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC1IlS1_ILl1ELl1000000000EEvEERKNS0_IT_T0_EE, @function
_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC1IlS1_ILl1ELl1000000000EEvEERKNS0_IT_T0_EE:
.LFB2022:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 32
	mov	QWORD PTR -24[rbp], rdi
	mov	QWORD PTR -32[rbp], rsi
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR -8[rbp], rax
	xor	eax, eax
	mov	rax, QWORD PTR -32[rbp]
	mov	rdi, rax
	call	_ZNSt6chrono13duration_castINS_8durationIdSt5ratioILl1ELl1EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE
	movq	rax, xmm0
	movq	xmm0, rax
	movsd	QWORD PTR -16[rbp], xmm0
	lea	rax, -16[rbp]
	mov	rdi, rax
	call	_ZNKSt6chrono8durationIdSt5ratioILl1ELl1EEE5countEv
	movq	rax, xmm0
	mov	rdx, QWORD PTR -24[rbp]
	mov	QWORD PTR [rdx], rax
	nop
	mov	rax, QWORD PTR -8[rbp]
	sub	rax, QWORD PTR fs:40
	je	.L5
	call	__stack_chk_fail@PLT
.L5:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2022:
	.size	_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC1IlS1_ILl1ELl1000000000EEvEERKNS0_IT_T0_EE, .-_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC1IlS1_ILl1ELl1000000000EEvEERKNS0_IT_T0_EE
	.section	.text._ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE,"axG",@progbits,_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE,comdat
	.weak	_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE
	.type	_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE, @function
_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE:
.LFB2023:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 48
	mov	QWORD PTR -40[rbp], rdi
	mov	QWORD PTR -48[rbp], rsi
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR -8[rbp], rax
	xor	eax, eax
	mov	rax, QWORD PTR -48[rbp]
	mov	rdi, rax
	call	_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv
	mov	QWORD PTR -16[rbp], rax
	mov	rax, QWORD PTR -40[rbp]
	mov	rdi, rax
	call	_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv
	mov	QWORD PTR -24[rbp], rax
	lea	rdx, -16[rbp]
	lea	rax, -24[rbp]
	mov	rsi, rdx
	mov	rdi, rax
	call	_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_
	mov	rdx, QWORD PTR -8[rbp]
	sub	rdx, QWORD PTR fs:40
	je	.L8
	call	__stack_chk_fail@PLT
.L8:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2023:
	.size	_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE, .-_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE
	.section	.text._ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_,"axG",@progbits,_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_,comdat
	.weak	_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_
	.type	_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_, @function
_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_:
.LFB2024:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	push	rbx
	sub	rsp, 72
	.cfi_offset 3, -24
	mov	QWORD PTR -72[rbp], rdi
	mov	QWORD PTR -80[rbp], rsi
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR -24[rbp], rax
	xor	eax, eax
	mov	rax, QWORD PTR -72[rbp]
	mov	rax, QWORD PTR [rax]
	mov	QWORD PTR -56[rbp], rax
	lea	rax, -56[rbp]
	mov	rdi, rax
	call	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv
	mov	rbx, rax
	mov	rax, QWORD PTR -80[rbp]
	mov	rax, QWORD PTR [rax]
	mov	QWORD PTR -48[rbp], rax
	lea	rax, -48[rbp]
	mov	rdi, rax
	call	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv
	mov	rdx, rax
	mov	rax, rbx
	sub	rax, rdx
	mov	QWORD PTR -40[rbp], rax
	lea	rdx, -40[rbp]
	lea	rax, -32[rbp]
	mov	rsi, rdx
	mov	rdi, rax
	call	_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_
	mov	rax, QWORD PTR -32[rbp]
	mov	rdx, QWORD PTR -24[rbp]
	sub	rdx, QWORD PTR fs:40
	je	.L11
	call	__stack_chk_fail@PLT
.L11:
	mov	rbx, QWORD PTR -8[rbp]
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2024:
	.size	_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_, .-_ZNSt6chronomiIlSt5ratioILl1ELl1000000000EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_
	.section	.text._ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv,"axG",@progbits,_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv,comdat
	.align 2
	.weak	_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv
	.type	_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv, @function
_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv:
.LFB2025:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	mov	QWORD PTR -8[rbp], rdi
	mov	rax, QWORD PTR -8[rbp]
	mov	rax, QWORD PTR [rax]
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2025:
	.size	_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv, .-_ZNKSt6chrono10time_pointINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEE16time_since_epochEv
	.section	.rodata
	.align 8
.LC2:
	.string	"Time taken(naive-ijk loop with  -O1 optimization flag in compilation flag) for the config  (M=2000,N=2000,K=2000) and with  no temporary sum_variable(to avoid global ram writes for every accumulation) for accumulations   : "
.LC3:
	.string	"secs"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2018:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 144
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR -8[rbp], rax
	xor	eax, eax
	mov	DWORD PTR -104[rbp], 2000
	mov	DWORD PTR -100[rbp], 2000
	mov	DWORD PTR -96[rbp], 2000
	mov	eax, DWORD PTR -104[rbp]
	imul	eax, DWORD PTR -96[rbp]
	cdqe
	sal	eax, 2
	mov	DWORD PTR -92[rbp], eax
	mov	eax, DWORD PTR -96[rbp]
	imul	eax, DWORD PTR -100[rbp]
	cdqe
	sal	eax, 2
	mov	DWORD PTR -88[rbp], eax
	mov	eax, DWORD PTR -104[rbp]
	imul	eax, DWORD PTR -100[rbp]
	cdqe
	sal	eax, 2
	mov	DWORD PTR -84[rbp], eax
	mov	eax, DWORD PTR -104[rbp]
	imul	eax, DWORD PTR -96[rbp]
	cdqe
	movabs	rdx, 2305843009213693950
	cmp	rax, rdx
	ja	.L15
	sal	rax, 2
	mov	rdi, rax
	call	_Znam@PLT
	mov	QWORD PTR -32[rbp], rax
	mov	eax, DWORD PTR -96[rbp]
	imul	eax, DWORD PTR -100[rbp]
	cdqe
	movabs	rdx, 2305843009213693950
	cmp	rax, rdx
	ja	.L17
	jmp	.L46
.L15:
	call	__cxa_throw_bad_array_new_length@PLT
.L46:
	sal	rax, 2
	mov	rdi, rax
	call	_Znam@PLT
	mov	QWORD PTR -24[rbp], rax
	mov	eax, DWORD PTR -104[rbp]
	imul	eax, DWORD PTR -100[rbp]
	cdqe
	movabs	rdx, 2305843009213693950
	cmp	rax, rdx
	ja	.L19
	jmp	.L47
.L17:
	call	__cxa_throw_bad_array_new_length@PLT
.L47:
	sal	rax, 2
	mov	rdi, rax
	call	_Znam@PLT
	mov	QWORD PTR -16[rbp], rax
	mov	DWORD PTR -144[rbp], 0
	jmp	.L21
.L19:
	call	__cxa_throw_bad_array_new_length@PLT
.L22:
	mov	eax, DWORD PTR -144[rbp]
	cdqe
	lea	rdx, 0[0+rax*4]
	mov	rax, QWORD PTR -32[rbp]
	add	rax, rdx
	movss	xmm0, DWORD PTR .LC0[rip]
	movss	DWORD PTR [rax], xmm0
	add	DWORD PTR -144[rbp], 1
.L21:
	mov	eax, DWORD PTR -104[rbp]
	imul	eax, DWORD PTR -96[rbp]
	cmp	DWORD PTR -144[rbp], eax
	jl	.L22
	mov	DWORD PTR -140[rbp], 0
	jmp	.L23
.L24:
	mov	eax, DWORD PTR -140[rbp]
	cdqe
	lea	rdx, 0[0+rax*4]
	mov	rax, QWORD PTR -24[rbp]
	add	rax, rdx
	movss	xmm0, DWORD PTR .LC0[rip]
	movss	DWORD PTR [rax], xmm0
	add	DWORD PTR -140[rbp], 1
.L23:
	mov	eax, DWORD PTR -96[rbp]
	imul	eax, DWORD PTR -100[rbp]
	cmp	DWORD PTR -140[rbp], eax
	jl	.L24
	pxor	xmm0, xmm0
	movsd	QWORD PTR -40[rbp], xmm0
	mov	DWORD PTR -80[rbp], 10
	mov	DWORD PTR -76[rbp], 3
	mov	DWORD PTR -136[rbp], 0
	jmp	.L25
.L32:
	mov	eax, DWORD PTR -84[rbp]
	movsx	rdx, eax
	mov	rax, QWORD PTR -16[rbp]
	mov	esi, 0
	mov	rdi, rax
	call	memset@PLT
	mov	DWORD PTR -132[rbp], 0
	jmp	.L26
.L31:
	mov	DWORD PTR -128[rbp], 0
	jmp	.L27
.L30:
	mov	DWORD PTR -124[rbp], 0
	jmp	.L28
.L29:
	mov	eax, DWORD PTR -132[rbp]
	imul	eax, DWORD PTR -100[rbp]
	mov	edx, eax
	mov	eax, DWORD PTR -128[rbp]
	add	eax, edx
	cdqe
	lea	rdx, 0[0+rax*4]
	mov	rax, QWORD PTR -16[rbp]
	add	rax, rdx
	movss	xmm1, DWORD PTR [rax]
	mov	eax, DWORD PTR -132[rbp]
	imul	eax, DWORD PTR -96[rbp]
	mov	edx, eax
	mov	eax, DWORD PTR -124[rbp]
	add	eax, edx
	cdqe
	lea	rdx, 0[0+rax*4]
	mov	rax, QWORD PTR -32[rbp]
	add	rax, rdx
	movss	xmm2, DWORD PTR [rax]
	mov	eax, DWORD PTR -124[rbp]
	imul	eax, DWORD PTR -100[rbp]
	mov	edx, eax
	mov	eax, DWORD PTR -128[rbp]
	add	eax, edx
	cdqe
	lea	rdx, 0[0+rax*4]
	mov	rax, QWORD PTR -24[rbp]
	add	rax, rdx
	movss	xmm0, DWORD PTR [rax]
	mulss	xmm0, xmm2
	mov	eax, DWORD PTR -132[rbp]
	imul	eax, DWORD PTR -100[rbp]
	mov	edx, eax
	mov	eax, DWORD PTR -128[rbp]
	add	eax, edx
	cdqe
	lea	rdx, 0[0+rax*4]
	mov	rax, QWORD PTR -16[rbp]
	add	rax, rdx
	addss	xmm0, xmm1
	movss	DWORD PTR [rax], xmm0
	add	DWORD PTR -124[rbp], 1
.L28:
	mov	eax, DWORD PTR -124[rbp]
	cmp	eax, DWORD PTR -96[rbp]
	jl	.L29
	add	DWORD PTR -128[rbp], 1
.L27:
	mov	eax, DWORD PTR -128[rbp]
	cmp	eax, DWORD PTR -100[rbp]
	jl	.L30
	add	DWORD PTR -132[rbp], 1
.L26:
	mov	eax, DWORD PTR -132[rbp]
	cmp	eax, DWORD PTR -104[rbp]
	jl	.L31
	add	DWORD PTR -136[rbp], 1
.L25:
	mov	eax, DWORD PTR -136[rbp]
	cmp	eax, DWORD PTR -76[rbp]
	jl	.L32
	mov	DWORD PTR -120[rbp], 0
	jmp	.L33
.L40:
	mov	eax, DWORD PTR -84[rbp]
	movsx	rdx, eax
	mov	rax, QWORD PTR -16[rbp]
	mov	esi, 0
	mov	rdi, rax
	call	memset@PLT
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	mov	QWORD PTR -72[rbp], rax
	mov	DWORD PTR -116[rbp], 0
	jmp	.L34
.L39:
	mov	DWORD PTR -112[rbp], 0
	jmp	.L35
.L38:
	mov	DWORD PTR -108[rbp], 0
	jmp	.L36
.L37:
	mov	eax, DWORD PTR -116[rbp]
	imul	eax, DWORD PTR -100[rbp]
	mov	edx, eax
	mov	eax, DWORD PTR -112[rbp]
	add	eax, edx
	cdqe
	lea	rdx, 0[0+rax*4]
	mov	rax, QWORD PTR -16[rbp]
	add	rax, rdx
	movss	xmm1, DWORD PTR [rax]
	mov	eax, DWORD PTR -116[rbp]
	imul	eax, DWORD PTR -96[rbp]
	mov	edx, eax
	mov	eax, DWORD PTR -108[rbp]
	add	eax, edx
	cdqe
	lea	rdx, 0[0+rax*4]
	mov	rax, QWORD PTR -32[rbp]
	add	rax, rdx
	movss	xmm2, DWORD PTR [rax]
	mov	eax, DWORD PTR -108[rbp]
	imul	eax, DWORD PTR -100[rbp]
	mov	edx, eax
	mov	eax, DWORD PTR -112[rbp]
	add	eax, edx
	cdqe
	lea	rdx, 0[0+rax*4]
	mov	rax, QWORD PTR -24[rbp]
	add	rax, rdx
	movss	xmm0, DWORD PTR [rax]
	mulss	xmm0, xmm2
	mov	eax, DWORD PTR -116[rbp]
	imul	eax, DWORD PTR -100[rbp]
	mov	edx, eax
	mov	eax, DWORD PTR -112[rbp]
	add	eax, edx
	cdqe
	lea	rdx, 0[0+rax*4]
	mov	rax, QWORD PTR -16[rbp]
	add	rax, rdx
	addss	xmm0, xmm1
	movss	DWORD PTR [rax], xmm0
	add	DWORD PTR -108[rbp], 1
.L36:
	mov	eax, DWORD PTR -108[rbp]
	cmp	eax, DWORD PTR -96[rbp]
	jl	.L37
	add	DWORD PTR -112[rbp], 1
.L35:
	mov	eax, DWORD PTR -112[rbp]
	cmp	eax, DWORD PTR -100[rbp]
	jl	.L38
	add	DWORD PTR -116[rbp], 1
.L34:
	mov	eax, DWORD PTR -116[rbp]
	cmp	eax, DWORD PTR -104[rbp]
	jl	.L39
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	mov	QWORD PTR -64[rbp], rax
	lea	rdx, -72[rbp]
	lea	rax, -64[rbp]
	mov	rsi, rdx
	mov	rdi, rax
	call	_ZNSt6chronomiINS_3_V212system_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEES6_EENSt11common_typeIJT0_T1_EE4typeERKNS_10time_pointIT_S8_EERKNSC_ISD_S9_EE
	mov	QWORD PTR -48[rbp], rax
	lea	rdx, -48[rbp]
	lea	rax, -56[rbp]
	mov	rsi, rdx
	mov	rdi, rax
	call	_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC1IlS1_ILl1ELl1000000000EEvEERKNS0_IT_T0_EE
	lea	rax, -56[rbp]
	mov	rdi, rax
	call	_ZNKSt6chrono8durationIdSt5ratioILl1ELl1EEE5countEv
	movsd	xmm1, QWORD PTR -40[rbp]
	addsd	xmm0, xmm1
	movsd	QWORD PTR -40[rbp], xmm0
	add	DWORD PTR -120[rbp], 1
.L33:
	mov	eax, DWORD PTR -120[rbp]
	cmp	eax, DWORD PTR -80[rbp]
	jl	.L40
	pxor	xmm1, xmm1
	cvtsi2sd	xmm1, DWORD PTR -80[rbp]
	movsd	xmm0, QWORD PTR -40[rbp]
	divsd	xmm0, xmm1
	movsd	QWORD PTR -40[rbp], xmm0
	lea	rax, .LC2[rip]
	mov	rsi, rax
	lea	rax, _ZSt4cout[rip]
	mov	rdi, rax
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	mov	rdx, rax
	mov	rax, QWORD PTR -40[rbp]
	movq	xmm0, rax
	mov	rdi, rdx
	call	_ZNSolsEd@PLT
	mov	rdx, rax
	lea	rax, .LC3[rip]
	mov	rsi, rax
	mov	rdi, rdx
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	mov	rdx, QWORD PTR _ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_@GOTPCREL[rip]
	mov	rsi, rdx
	mov	rdi, rax
	call	_ZNSolsEPFRSoS_E@PLT
	cmp	QWORD PTR -32[rbp], 0
	je	.L41
	mov	rax, QWORD PTR -32[rbp]
	mov	rdi, rax
	call	_ZdaPv@PLT
.L41:
	cmp	QWORD PTR -24[rbp], 0
	je	.L42
	mov	rax, QWORD PTR -24[rbp]
	mov	rdi, rax
	call	_ZdaPv@PLT
.L42:
	cmp	QWORD PTR -16[rbp], 0
	je	.L43
	mov	rax, QWORD PTR -16[rbp]
	mov	rdi, rax
	call	_ZdaPv@PLT
.L43:
	mov	eax, 0
	mov	rdx, QWORD PTR -8[rbp]
	sub	rdx, QWORD PTR fs:40
	je	.L45
	call	__stack_chk_fail@PLT
.L45:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2018:
	.size	main, .-main
	.section	.text._ZNSt6chrono13duration_castINS_8durationIdSt5ratioILl1ELl1EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE,"axG",@progbits,_ZNSt6chrono13duration_castINS_8durationIdSt5ratioILl1ELl1EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE,comdat
	.weak	_ZNSt6chrono13duration_castINS_8durationIdSt5ratioILl1ELl1EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE
	.type	_ZNSt6chrono13duration_castINS_8durationIdSt5ratioILl1ELl1EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE, @function
_ZNSt6chrono13duration_castINS_8durationIdSt5ratioILl1ELl1EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE:
.LFB2312:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 16
	mov	QWORD PTR -8[rbp], rdi
	mov	rax, QWORD PTR -8[rbp]
	mov	rdi, rax
	call	_ZNSt6chrono20__duration_cast_implINS_8durationIdSt5ratioILl1ELl1EEEES2_ILl1ELl1000000000EEdLb1ELb0EE6__castIlS5_EES4_RKNS1_IT_T0_EE
	movq	rax, xmm0
	movq	xmm0, rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2312:
	.size	_ZNSt6chrono13duration_castINS_8durationIdSt5ratioILl1ELl1EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE, .-_ZNSt6chrono13duration_castINS_8durationIdSt5ratioILl1ELl1EEEElS2_ILl1ELl1000000000EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE
	.section	.text._ZNKSt6chrono8durationIdSt5ratioILl1ELl1EEE5countEv,"axG",@progbits,_ZNKSt6chrono8durationIdSt5ratioILl1ELl1EEE5countEv,comdat
	.align 2
	.weak	_ZNKSt6chrono8durationIdSt5ratioILl1ELl1EEE5countEv
	.type	_ZNKSt6chrono8durationIdSt5ratioILl1ELl1EEE5countEv, @function
_ZNKSt6chrono8durationIdSt5ratioILl1ELl1EEE5countEv:
.LFB2313:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	mov	QWORD PTR -8[rbp], rdi
	mov	rax, QWORD PTR -8[rbp]
	movsd	xmm0, QWORD PTR [rax]
	movq	rax, xmm0
	movq	xmm0, rax
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2313:
	.size	_ZNKSt6chrono8durationIdSt5ratioILl1ELl1EEE5countEv, .-_ZNKSt6chrono8durationIdSt5ratioILl1ELl1EEE5countEv
	.section	.text._ZNSt6chrono20__duration_cast_implINS_8durationIdSt5ratioILl1ELl1EEEES2_ILl1ELl1000000000EEdLb1ELb0EE6__castIlS5_EES4_RKNS1_IT_T0_EE,"axG",@progbits,_ZNSt6chrono20__duration_cast_implINS_8durationIdSt5ratioILl1ELl1EEEES2_ILl1ELl1000000000EEdLb1ELb0EE6__castIlS5_EES4_RKNS1_IT_T0_EE,comdat
	.weak	_ZNSt6chrono20__duration_cast_implINS_8durationIdSt5ratioILl1ELl1EEEES2_ILl1ELl1000000000EEdLb1ELb0EE6__castIlS5_EES4_RKNS1_IT_T0_EE
	.type	_ZNSt6chrono20__duration_cast_implINS_8durationIdSt5ratioILl1ELl1EEEES2_ILl1ELl1000000000EEdLb1ELb0EE6__castIlS5_EES4_RKNS1_IT_T0_EE, @function
_ZNSt6chrono20__duration_cast_implINS_8durationIdSt5ratioILl1ELl1EEEES2_ILl1ELl1000000000EEdLb1ELb0EE6__castIlS5_EES4_RKNS1_IT_T0_EE:
.LFB2405:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 48
	mov	QWORD PTR -40[rbp], rdi
	mov	rax, QWORD PTR fs:40
	mov	QWORD PTR -8[rbp], rax
	xor	eax, eax
	mov	rax, QWORD PTR -40[rbp]
	mov	rdi, rax
	call	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv
	pxor	xmm0, xmm0
	cvtsi2sd	xmm0, rax
	movsd	xmm1, QWORD PTR .LC4[rip]
	divsd	xmm0, xmm1
	movsd	QWORD PTR -24[rbp], xmm0
	lea	rdx, -24[rbp]
	lea	rax, -16[rbp]
	mov	rsi, rdx
	mov	rdi, rax
	call	_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC1IdvEERKT_
	movsd	xmm0, QWORD PTR -16[rbp]
	movq	rax, xmm0
	mov	rdx, QWORD PTR -8[rbp]
	sub	rdx, QWORD PTR fs:40
	je	.L54
	call	__stack_chk_fail@PLT
.L54:
	movq	xmm0, rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2405:
	.size	_ZNSt6chrono20__duration_cast_implINS_8durationIdSt5ratioILl1ELl1EEEES2_ILl1ELl1000000000EEdLb1ELb0EE6__castIlS5_EES4_RKNS1_IT_T0_EE, .-_ZNSt6chrono20__duration_cast_implINS_8durationIdSt5ratioILl1ELl1EEEES2_ILl1ELl1000000000EEdLb1ELb0EE6__castIlS5_EES4_RKNS1_IT_T0_EE
	.section	.text._ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC2IdvEERKT_,"axG",@progbits,_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC5IdvEERKT_,comdat
	.align 2
	.weak	_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC2IdvEERKT_
	.type	_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC2IdvEERKT_, @function
_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC2IdvEERKT_:
.LFB2447:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	mov	QWORD PTR -8[rbp], rdi
	mov	QWORD PTR -16[rbp], rsi
	mov	rax, QWORD PTR -16[rbp]
	movsd	xmm0, QWORD PTR [rax]
	mov	rax, QWORD PTR -8[rbp]
	movsd	QWORD PTR [rax], xmm0
	nop
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2447:
	.size	_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC2IdvEERKT_, .-_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC2IdvEERKT_
	.weak	_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC1IdvEERKT_
	.set	_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC1IdvEERKT_,_ZNSt6chrono8durationIdSt5ratioILl1ELl1EEEC2IdvEERKT_
	.text
	.type	_Z41__static_initialization_and_destruction_0ii, @function
_Z41__static_initialization_and_destruction_0ii:
.LFB2577:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	sub	rsp, 16
	mov	DWORD PTR -4[rbp], edi
	mov	DWORD PTR -8[rbp], esi
	cmp	DWORD PTR -4[rbp], 1
	jne	.L58
	cmp	DWORD PTR -8[rbp], 65535
	jne	.L58
	lea	rax, _ZStL8__ioinit[rip]
	mov	rdi, rax
	call	_ZNSt8ios_base4InitC1Ev@PLT
	lea	rax, __dso_handle[rip]
	mov	rdx, rax
	lea	rax, _ZStL8__ioinit[rip]
	mov	rsi, rax
	mov	rax, QWORD PTR _ZNSt8ios_base4InitD1Ev@GOTPCREL[rip]
	mov	rdi, rax
	call	__cxa_atexit@PLT
.L58:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2577:
	.size	_Z41__static_initialization_and_destruction_0ii, .-_Z41__static_initialization_and_destruction_0ii
	.type	_GLOBAL__sub_I_main, @function
_GLOBAL__sub_I_main:
.LFB2578:
	.cfi_startproc
	endbr64
	push	rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	mov	rbp, rsp
	.cfi_def_cfa_register 6
	mov	esi, 65535
	mov	edi, 1
	call	_Z41__static_initialization_and_destruction_0ii
	pop	rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2578:
	.size	_GLOBAL__sub_I_main, .-_GLOBAL__sub_I_main
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I_main
	.section	.rodata
	.align 4
.LC0:
	.long	1065353216
	.align 8
.LC4:
	.long	0
	.long	1104006501
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
