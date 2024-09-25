#pragma once

#include <assert.h>
#include <memory>
#include <vector>
#include <complex>

namespace chase
{
namespace linalg
{
namespace scalapackpp
{
void t_descinit(std::size_t* desc, std::size_t* m, std::size_t* n,
                std::size_t* mb, std::size_t* nb, int* irsrc, int* icsrc,
                int* ictxt, std::size_t* lld, int* info)
{
    int* desc_ = new int[9];
    int m_ = *m;
    int n_ = *n;
    int mb_ = *mb;
    int nb_ = *nb;
    int irsrc_ = *irsrc;
    int icsrc_ = *icsrc;
    int ictxt_ = *ictxt;
    int lld_ = *lld;
    int info_ = *info;
    descinit_(desc_, &m_, &n_, &mb_, &nb_, &irsrc_, &icsrc_, &ictxt_, &lld_,
              &info_);
    *info = info_;

    for (int i = 0; i < 9; i++)
    {
        desc[i] = static_cast<std::size_t>(desc_[i]);
    }
}

template <>
void t_pgeqrf(std::size_t m, std::size_t n, double* A, int ia, int ja,
              std::size_t* desc_a, double* tau)
{
    int* desc_ = new int[9];
    for (int i = 0; i < 9; i++)
    {
        desc_[i] = static_cast<std::size_t>(desc_a[i]);
    }
    int m_ = m;
    int n_ = n;

    double* work;
    double numwork;
    int lwork, info;
    lwork = -1;

    pdgeqrf_(&m_, &n_, A, &ia, &ja, desc_, tau, &numwork, &lwork, &info);

    lwork = (int)numwork;

    auto wptr = std::unique_ptr<double[]>{new double[lwork]};

    work = wptr.get();

    pdgeqrf_(&m_, &n_, A, &ia, &ja, desc_, tau, work, &lwork, &info);
    assert(info == 0);
}

template <>
void t_pgeqrf(std::size_t m, std::size_t n, float* A, int ia, int ja,
              std::size_t* desc_a, float* tau)
{
    int* desc_ = new int[9];
    for (int i = 0; i < 9; i++)
    {
        desc_[i] = static_cast<std::size_t>(desc_a[i]);
    }
    int m_ = m;
    int n_ = n;

    float* work;
    float numwork;
    int lwork, info;
    lwork = -1;

    psgeqrf_(&m_, &n_, A, &ia, &ja, desc_, tau, &numwork, &lwork, &info);

    lwork = (int)numwork;

    auto wptr = std::unique_ptr<float[]>{new float[lwork]};

    work = wptr.get();

    psgeqrf_(&m_, &n_, A, &ia, &ja, desc_, tau, work, &lwork, &info);
    assert(info == 0);
}

template <>
void t_pgeqrf(std::size_t m, std::size_t n, std::complex<double>* A, int ia,
              int ja, std::size_t* desc_a, std::complex<double>* tau)
{
    int* desc_ = new int[9];
    for (int i = 0; i < 9; i++)
    {
        desc_[i] = static_cast<std::size_t>(desc_a[i]);
    }
    int m_ = m;
    int n_ = n;

    std::complex<double>* work;
    std::complex<double> numwork;
    int lwork, info;
    lwork = -1;

    pzgeqrf_(&m_, &n_, A, &ia, &ja, desc_, tau, &numwork, &lwork, &info);

    lwork = (int)real(numwork);

    auto wptr = std::unique_ptr<std::complex<double>[]> {
        new std::complex<double>[ lwork ]
    };

    work = wptr.get();

    pzgeqrf_(&m_, &n_, A, &ia, &ja, desc_, tau, work, &lwork, &info);
    assert(info == 0);
}

template <>
void t_pgeqrf(std::size_t m, std::size_t n, std::complex<float>* A, int ia,
              int ja, std::size_t* desc_a, std::complex<float>* tau)
{
    int* desc_ = new int[9];
    for (int i = 0; i < 9; i++)
    {
        desc_[i] = static_cast<std::size_t>(desc_a[i]);
    }
    int m_ = m;
    int n_ = n;

    std::complex<float>* work;
    std::complex<float> numwork;
    int lwork, info;
    lwork = -1;

    pcgeqrf_(&m_, &n_, A, &ia, &ja, desc_, tau, &numwork, &lwork, &info);

    lwork = (int)real(numwork);

    auto wptr = std::unique_ptr<std::complex<float>[]> {
        new std::complex<float>[ lwork ]
    };

    work = wptr.get();

    pcgeqrf_(&m_, &n_, A, &ia, &ja, desc_, tau, work, &lwork, &info);
    assert(info == 0);
}

template <>
void t_pgqr(std::size_t m, std::size_t n, std::size_t k, double* A, int ia,
            int ja, std::size_t* desc_a, double* tau)
{
    int* desc_ = new int[9];
    for (int i = 0; i < 9; i++)
    {
        desc_[i] = static_cast<std::size_t>(desc_a[i]);
    }
    int m_ = m;
    int n_ = n;
    int k_ = k;

    double* work;
    double numwork;
    int lwork, info;
    lwork = -1;

    pdorgqr_(&m_, &n_, &k_, A, &ia, &ja, desc_, tau, &numwork, &lwork, &info);

    lwork = (int)numwork;

    auto wptr = std::unique_ptr<double[]>{new double[lwork]};

    work = wptr.get();

    pdorgqr_(&m_, &n_, &k_, A, &ia, &ja, desc_, tau, work, &lwork, &info);
    assert(info == 0);
}

template <>
void t_pgqr(std::size_t m, std::size_t n, std::size_t k, float* A, int ia,
            int ja, std::size_t* desc_a, float* tau)
{
    int* desc_ = new int[9];
    for (int i = 0; i < 9; i++)
    {
        desc_[i] = static_cast<std::size_t>(desc_a[i]);
    }
    int m_ = m;
    int n_ = n;
    int k_ = n;

    float* work;
    float numwork;
    int lwork, info;
    lwork = -1;

    psorgqr_(&m_, &n_, &k_, A, &ia, &ja, desc_, tau, &numwork, &lwork, &info);

    lwork = (int)numwork;

    auto wptr = std::unique_ptr<float[]>{new float[lwork]};

    work = wptr.get();

    psorgqr_(&m_, &n_, &k_, A, &ia, &ja, desc_, tau, work, &lwork, &info);
    assert(info == 0);
}

template <>
void t_pgqr(std::size_t m, std::size_t n, std::size_t k,
            std::complex<double>* A, int ia, int ja, std::size_t* desc_a,
            std::complex<double>* tau)
{
    int* desc_ = new int[9];
    for (int i = 0; i < 9; i++)
    {
        desc_[i] = static_cast<std::size_t>(desc_a[i]);
    }
    int m_ = m;
    int n_ = n;
    int k_ = k;

    std::complex<double>* work;
    std::complex<double> numwork;
    int lwork, info;
    lwork = -1;

    pzungqr_(&m_, &n_, &k_, A, &ia, &ja, desc_, tau, &numwork, &lwork, &info);

    lwork = (int)real(numwork);

    auto wptr = std::unique_ptr<std::complex<double>[]> {
        new std::complex<double>[ lwork ]
    };

    work = wptr.get();

    pzungqr_(&m_, &n_, &k_, A, &ia, &ja, desc_, tau, work, &lwork, &info);
    assert(info == 0);
}

template <>
void t_pgqr(std::size_t m, std::size_t n, std::size_t k, std::complex<float>* A,
            int ia, int ja, std::size_t* desc_a, std::complex<float>* tau)
{
    int* desc_ = new int[9];
    for (int i = 0; i < 9; i++)
    {
        desc_[i] = static_cast<std::size_t>(desc_a[i]);
    }
    int m_ = m;
    int n_ = n;
    int k_ = k;

    std::complex<float>* work;
    std::complex<float> numwork;
    int lwork, info;
    lwork = -1;

    pcungqr_(&m_, &n_, &k_, A, &ia, &ja, desc_, tau, &numwork, &lwork, &info);

    lwork = (int)real(numwork);

    auto wptr = std::unique_ptr<std::complex<float>[]> {
        new std::complex<float>[ lwork ]
    };

    work = wptr.get();

    pcungqr_(&m_, &n_, &k_, A, &ia, &ja, desc_, tau, work, &lwork, &info);
    assert(info == 0);
}


template <>
void t_ptranc(std::size_t m, std::size_t n, float alpha, float *A, int ia, int ja,
            std::size_t* desc_a, float beta, float *C, int ic, int jc, std::size_t* desc_c)
{
    int* desc_a_ = new int[9];
    int* desc_c_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<std::size_t>(desc_a[i]);
        desc_c_[i] = static_cast<std::size_t>(desc_c[i]);
    }
    int m_ = m;
    int n_ = n;   

    pstran_(&m_, &n_, &alpha, A, &ia, &ja, desc_a_, &beta, C, &ic, &jc, desc_c_); 
}

template <>
void t_ptranc(std::size_t m, std::size_t n, double alpha, double *A, int ia, int ja,
            std::size_t* desc_a, double beta, double *C, int ic, int jc, std::size_t* desc_c)
{
    int* desc_a_ = new int[9];
    int* desc_c_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<std::size_t>(desc_a[i]);
        desc_c_[i] = static_cast<std::size_t>(desc_c[i]);
    }
    int m_ = m;
    int n_ = n;   

    pdtran_(&m_, &n_, &alpha, A, &ia, &ja, desc_a_, &beta, C, &ic, &jc, desc_c_); 
}

template <>
void t_ptranc(std::size_t m, std::size_t n, std::complex<float> alpha, std::complex<float> *A, int ia, int ja,
            std::size_t* desc_a, std::complex<float> beta, std::complex<float> *C, int ic, int jc, std::size_t* desc_c)
{
    int* desc_a_ = new int[9];
    int* desc_c_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<std::size_t>(desc_a[i]);
        desc_c_[i] = static_cast<std::size_t>(desc_c[i]);
    }
    int m_ = m;
    int n_ = n;   

    pctranc_(&m_, &n_, &alpha, A, &ia, &ja, desc_a_, &beta, C, &ic, &jc, desc_c_); 
}

template <>
void t_ptranc(std::size_t m, std::size_t n, std::complex<double> alpha, std::complex<double> *A, int ia, int ja,
            std::size_t* desc_a, std::complex<double> beta, std::complex<double> *C, int ic, int jc, std::size_t* desc_c)
{
    int* desc_a_ = new int[9];
    int* desc_c_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<std::size_t>(desc_a[i]);
        desc_c_[i] = static_cast<std::size_t>(desc_c[i]);
    }
    int m_ = m;
    int n_ = n;   

    pztranc_(&m_, &n_, &alpha, A, &ia, &ja, desc_a_, &beta, C, &ic, &jc, desc_c_); 
}

}
}
}