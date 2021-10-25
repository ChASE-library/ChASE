/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <fortran_mangle.h>
#include <complex>

namespace chase {
namespace mpi {
using BlasInt = int;
using dcomplex = std::complex<double>;
using scomplex = std::complex<float>;

extern "C" {

////////////
// BLACS //
////////////
void FC_GLOBAL(blacs_pinfo, BLACS_PINFO)(BlasInt *mypnum, BlasInt *nprocs);
void FC_GLOBAL(blacs_get, BLACS_GET)(BlasInt *icontxt, const BlasInt *what, BlasInt *val );
void FC_GLOBAL(blacs_gridinit, BLACS_GRIDINIT)(BlasInt *icontxt, const char* layout, 
				const BlasInt *nprow, const BlasInt *npcol);
void FC_GLOBAL(blacs_gridinfo, BLACS_GRIDINFO)(BlasInt *icontxt, BlasInt *nprow, BlasInt *npcol, 
			        BlasInt *myprow, BlasInt *mypcol);

int FC_GLOBAL(numroc, NUMROC)(BlasInt *n, BlasInt *nb, BlasInt *iproc, 
			      const BlasInt *isrcproc, BlasInt *nprocs);

void FC_GLOBAL(descinit, DESCINIT)(BlasInt *desc, BlasInt *m, BlasInt *n, BlasInt *mb, BlasInt *nb, 
		const BlasInt *irsrc, const BlasInt *icsrc, BlasInt *ictxt, BlasInt *lld, BlasInt *info);

///////////////
// Scalapack //
//////////////

//PBLAS
//psgeadd
void FC_GLOBAL(psgeadd, PSGEADD)(const char* trans, BlasInt *m, BlasInt *n, float* alpha, float* a,
	       			 BlasInt *ia, BlasInt *ja, BlasInt *desc_a, float* beta, float* c,  BlasInt *ic, 
				 BlasInt *jc, BlasInt *desc_c);
//pdgeadd
void FC_GLOBAL(pdgeadd, PDGEADD)(const char* trans, BlasInt *m, BlasInt *n, double* alpha, double* a,
                                 BlasInt *ia, BlasInt *ja, BlasInt *desc_a, double* beta, double* c,  BlasInt *ic,
                                 BlasInt *jc, BlasInt *desc_c);

//pcgeadd
void FC_GLOBAL(pcgeadd, PCGEADD)(const char* trans, BlasInt *m, BlasInt *n, std::complex<float>* alpha, std::complex<float>* a,
                                 BlasInt *ia, BlasInt *ja, BlasInt *desc_a, std::complex<float>* beta, std::complex<float>* c,  BlasInt *ic,
                                 BlasInt *jc, BlasInt *desc_c);

//pzgeadd
void FC_GLOBAL(pzgeadd, PZGEADD)(const char* trans, BlasInt *m, BlasInt *n, std::complex<double>* alpha, std::complex<double>* a,
                                 BlasInt *ia, BlasInt *ja, BlasInt *desc_a, std::complex<double>* beta, std::complex<double>* c,  BlasInt *ic,
                                 BlasInt *jc, BlasInt *desc_c);

//Cholesky factorization
//pspotrf
void FC_GLOBAL(pspotrf, PSPOTRF)(const char* uplo, BlasInt *n, float *a, BlasInt *ia,
			         BlasInt *ja, BlasInt *desc_a, BlasInt *info);
//pdpotrf
void FC_GLOBAL(pdpotrf, PDPOTRF)(const char* uplo, BlasInt *n, double *a, BlasInt *ia,
                                 BlasInt *ja, BlasInt *desc_a, BlasInt *info);
//pcpotrf
void FC_GLOBAL(pcpotrf, PCPOTRF)(const char* uplo, BlasInt *n, scomplex *a, BlasInt *ia,
                                 BlasInt *ja, BlasInt *desc_a, BlasInt *info);
//pzpotrf
void FC_GLOBAL(pzpotrf, PZPOTRF)(const char* uplo, BlasInt *n, dcomplex *a, BlasInt *ia,
                                 BlasInt *ja, BlasInt *desc_a, BlasInt *info);

//reduce generalized eigenproblem to a standard one
//pssygst
void FC_GLOBAL(pssygst, PSSYGST)(BlasInt *ibtype, const char* uplo, BlasInt *n, float *a,
				 BlasInt *ia, BlasInt *ja, BlasInt *desc_a, const float *b,
				 BlasInt *ib, BlasInt *jb, BlasInt *desc_b, float *scale,
				 BlasInt *info);

//pdsygst
void FC_GLOBAL(pdsygst, PDSYGST)(BlasInt *ibtype, const char* uplo, BlasInt *n, double *a,
                                 BlasInt *ia, BlasInt *ja, BlasInt *desc_a, const double *b,
                                 BlasInt *ib, BlasInt *jb, BlasInt *desc_b, double *scale,
                                 BlasInt *info);


//pchegst
void FC_GLOBAL(pchegst, PCHEGST)(BlasInt *ibtype, const char* uplo, BlasInt *n, scomplex *a,
                                 BlasInt *ia, BlasInt *ja, BlasInt *desc_a, const scomplex *b,
                                 BlasInt *ib, BlasInt *jb, BlasInt *desc_b, float *scale,
                                 BlasInt *info);
//pzhegst
void FC_GLOBAL(pzhegst, PZHEGST)(BlasInt *ibtype, const char* uplo, BlasInt *n, dcomplex *a,
                                 BlasInt *ia, BlasInt *ja, BlasInt *desc_a, const dcomplex *b,
                                 BlasInt *ib, BlasInt *jb, BlasInt *desc_b, double *scale,
                                 BlasInt *info);

//solve a triangular linear system
//pstrtrs
void FC_GLOBAL(pstrtrs, PSTRTRS)(const char* uplo, const char* trans, const char* diag,
				BlasInt *n, BlasInt *nhs, float *a,  BlasInt *ia, 
				BlasInt *ja, BlasInt *desc_a, float *b,  BlasInt *ib, 
				BlasInt *jb, BlasInt *desc_b, BlasInt *info);

//pdtrtrs
void FC_GLOBAL(pdtrtrs, PDTRTRS)(const char* uplo, const char* trans, const char* diag,
                                BlasInt *n, BlasInt *nhs, double *a,  BlasInt *ia,
                                BlasInt *ja, BlasInt *desc_a, double *b,  BlasInt *ib,
                                BlasInt *jb, BlasInt *desc_b, BlasInt *info);

//pctrtrs
void FC_GLOBAL(pctrtrs, PCTRTRS)(const char* uplo, const char* trans, const char* diag,
                                BlasInt *n, BlasInt *nhs, scomplex *a,  BlasInt *ia,
                                BlasInt *ja, BlasInt *desc_a, scomplex *b,  BlasInt *ib,
                                BlasInt *jb, BlasInt *desc_b, BlasInt *info);

//pztrtrs
void FC_GLOBAL(pztrtrs, PZTRTRS)(const char* uplo, const char* trans, const char* diag,
                                BlasInt *n, BlasInt *nhs, dcomplex *a,  BlasInt *ia,
                                BlasInt *ja, BlasInt *desc_a, dcomplex *b,  BlasInt *ib,
                                BlasInt *jb, BlasInt *desc_b, BlasInt *info);


}  // extern "C"
}  // namespace mpi
}  // namespace chase

