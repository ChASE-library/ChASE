#pragma once

#include <complex>

#include "algorithm/types.hpp"

#ifdef HAS_SCALAPACK
namespace chase
{
namespace linalg
{
namespace scalapackpp
{
extern "C" void blacs_get_(int*, int*, int*);
extern "C" void blacs_pinfo_(int*, int*);
extern "C" void blacs_gridinit_(int*, char*, int*, int*);
extern "C" void blacs_gridinfo_(int*, int*, int*, int*, int*);
extern "C" void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*,
                          int*);
extern "C" void blacs_gridexit_(int*);
extern "C" void blacs_gridmap_(int*, int*, int*, int*, int*);
extern "C" int numroc_(std::size_t*, std::size_t*, int*, int*, int*);

extern "C" void pdgeqrf_(int*, int*, double*, int*, int*, int*, double*,
                         double*, int*, int*);
extern "C" void psgeqrf_(int*, int*, float*, int*, int*, int*, float*, float*,
                         int*, int*);
extern "C" void pcgeqrf_(int*, int*, std::complex<float>*, int*, int*, int*,
                         std::complex<float>*, std::complex<float>*, int*,
                         int*);
extern "C" void pzgeqrf_(int*, int*, std::complex<double>*, int*, int*, int*,
                         std::complex<double>*, std::complex<double>*, int*,
                         int*);

extern "C" void pdorgqr_(int*, int*, int*, double*, int*, int*, int*, double*,
                         double*, int*, int*);
extern "C" void psorgqr_(int*, int*, int*, float*, int*, int*, int*, float*,
                         float*, int*, int*);
extern "C" void pzungqr_(int*, int*, int*, std::complex<double>*, int*, int*,
                         int*, std::complex<double>*, std::complex<double>*,
                         int*, int*);
extern "C" void pcungqr_(int*, int*, int*, std::complex<float>*, int*, int*,
                         int*, std::complex<float>*, std::complex<float>*, int*,
                         int*);

extern "C" void pstran_(int *, int *, float *, float *, int*, int*, int*, 
                        float *, float *, int*, int*, int*);  
extern "C" void pdtran_(int *, int *, double *, double *, int*, int*, int*, 
                        double *, double *, int*, int*, int*);  
extern "C" void pctranc_(int *, int *, std::complex<float> *, std::complex<float> *, int*, int*, int*, 
                        std::complex<float> *, std::complex<float> *, int*, int*, int*);  
extern "C" void pztranc_(int *, int *, std::complex<double> *, std::complex<double> *, int*, int*, int*, 
                        std::complex<double> *, std::complex<double> *, int*, int*, int*);  

extern "C" void pssyevd_(char *, char*, int *, float *, int *, int *, int *, float *, float *, int *, int *, int *, float *, int *, int *, int *, int *);                        
extern "C" void pdsyevd_(char *, char*, int *, double *, int *, int *, int *, double *, double *, int *, int *, int *, double *, int *, int *, int *, int *);                        
extern "C" void pcheevd_(char *, char*, int *, std::complex<float> *, int *, int *, int *, float *, std::complex<float> *, int *, int *, int *, std::complex<float> *, int *, float *, int *, int *, int *, int *);                        
extern "C" void pzheevd_(char *, char*, int *, std::complex<double> *, int *, int *, int *, double *, std::complex<double> *, int *, int *, int *, std::complex<double> *, int *, double *, int *, int *, int *, int *);                        

template <typename T>
void t_pgeqrf(std::size_t m, std::size_t n, T* A, int ia, int ja,
              std::size_t* desc_a, T* tau);
template <typename T>
void t_pgqr(std::size_t m, std::size_t n, std::size_t k, T* A, int ia, int ja,
            std::size_t* desc_a, T* tau);
            
template <typename T>
void t_ptranc(std::size_t m, std::size_t n, T alpha, T *A, int ia, int ja,
            std::size_t* desc_a, T beta, T *C, int ic, int jc, std::size_t* desc_c);

template <typename T>
void t_pheevd(char jobz, char uplo, std::size_t N, T *A, std::size_t *desc_a,
                                       chase::Base<T>* W, T *Z, std::size_t *desc_z, int *info);

}
}
}

#include "scalapackpp.inc"
#endif