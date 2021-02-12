#pragma once

#include <complex>
#include "algorithm/types.hpp"

namespace chase {
namespace mpi {

void blacs_pinfo(int *mypnum, int *nprocs);
void blacs_get(int *icontxt, const int *what, int *val );
void blacs_gridinit(int *icontxt, const char layout, const int *nprow, const int *npcol);
void blacs_gridinfo(int *icontxt, int *nprow, int *npcol, int *myprow, int *mypcol);
std::size_t numroc(std::size_t *n, std::size_t *nb, int *iproc, const int *isrcproc, int *nprocs);
void descinit(std::size_t *desc, std::size_t *m, std::size_t *n, std::size_t *mb, std::size_t *nb,
        const int *irsrc, const int *icsrc, int *ictxt, std::size_t *lld, int *info);

template <typename T>
void t_ppotrf(const char uplo, const std::size_t n, T *a, const std::size_t ia, 
	      const std::size_t ja, std::size_t *desc_a);

template <typename T>
void t_psyhegst(const int ibtype, const char uplo,const std::size_t n, T *a, const std::size_t ia,
                const std::size_t ja, std::size_t *desc_a, const T *b, const std::size_t ib,
		const std::size_t jb, std::size_t *desc_b, Base<T> *scale);


template <typename T>
void t_ptrtrs(const char uplo, const char trans, const char diag, const std::size_t n,
	      const std::size_t nhs, T *a,  const std::size_t ia, const std::size_t ja, 
	      std::size_t *desc_a, T *b, const std::size_t ib, const std::size_t jb, 
	      std::size_t *desc_b);


}  // namespace mpi
}  // namespace chase

#include "scalapack_templates.inc"


