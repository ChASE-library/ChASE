// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <assert.h>
#include <complex>
#include <memory>
#include <vector>

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

    auto wptr = std::unique_ptr<std::complex<double>[]>{
        new std::complex<double>[lwork]};

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

    auto wptr =
        std::unique_ptr<std::complex<float>[]>{new std::complex<float>[lwork]};

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

    auto wptr = std::unique_ptr<std::complex<double>[]>{
        new std::complex<double>[lwork]};

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

    auto wptr =
        std::unique_ptr<std::complex<float>[]>{new std::complex<float>[lwork]};

    work = wptr.get();

    pcungqr_(&m_, &n_, &k_, A, &ia, &ja, desc_, tau, work, &lwork, &info);
    assert(info == 0);
}

template <>
void t_ptranc(std::size_t m, std::size_t n, float alpha, float* A, int ia,
              int ja, std::size_t* desc_a, float beta, float* C, int ic, int jc,
              std::size_t* desc_c)
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

    pstran_(&m_, &n_, &alpha, A, &ia, &ja, desc_a_, &beta, C, &ic, &jc,
            desc_c_);
}

template <>
void t_ptranc(std::size_t m, std::size_t n, double alpha, double* A, int ia,
              int ja, std::size_t* desc_a, double beta, double* C, int ic,
              int jc, std::size_t* desc_c)
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

    pdtran_(&m_, &n_, &alpha, A, &ia, &ja, desc_a_, &beta, C, &ic, &jc,
            desc_c_);
}

template <>
void t_ptranc(std::size_t m, std::size_t n, std::complex<float> alpha,
              std::complex<float>* A, int ia, int ja, std::size_t* desc_a,
              std::complex<float> beta, std::complex<float>* C, int ic, int jc,
              std::size_t* desc_c)
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

    pctranc_(&m_, &n_, &alpha, A, &ia, &ja, desc_a_, &beta, C, &ic, &jc,
             desc_c_);
}

template <>
void t_ptranc(std::size_t m, std::size_t n, std::complex<double> alpha,
              std::complex<double>* A, int ia, int ja, std::size_t* desc_a,
              std::complex<double> beta, std::complex<double>* C, int ic,
              int jc, std::size_t* desc_c)
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

    pztranc_(&m_, &n_, &alpha, A, &ia, &ja, desc_a_, &beta, C, &ic, &jc,
             desc_c_);
}

template <>
void t_pheevd(char jobz, char uplo, std::size_t N, float* A,
              std::size_t* desc_a, float* W, float* Z, std::size_t* desc_z,
              int* info) // if seems that if jobz='N', return parameter error,
                         // need to check later
{
    int* desc_a_ = new int[9];
    int* desc_z_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<std::size_t>(desc_a[i]);
        desc_z_[i] = static_cast<std::size_t>(desc_z[i]);
    }

    int N_ = N;

    float* work;
    float work_q;
    int lwork = -1;

    int* iwork;
    int iwork_size;
    int liwork = 1;

    int one = 1;

    pssyevd_(&jobz, &uplo, &N_, A, &one, &one, desc_a_, W, Z, &one, &one,
             desc_z_, &work_q, &lwork, &iwork_size, &liwork, info);

    int work_size = int(work_q);
    int size =
        1.5 * work_size; // queried buffer size seems not large enough, don't
                         // know why. for now, amplified by a factor 1.5
    auto wptr_ = std::unique_ptr<float[]>{new float[size]};

    work = wptr_.get();

    auto iwptr_ = std::unique_ptr<int[]>{new int[iwork_size]};

    iwork = iwptr_.get();

    pssyevd_(&jobz, &uplo, &N_, A, &one, &one, desc_a_, W, Z, &one, &one,
             desc_z_, work, &size, iwork, &iwork_size, info);

    assert(*info == 0);
}

template <>
void t_pheevd(char jobz, char uplo, std::size_t N, double* A,
              std::size_t* desc_a, double* W, double* Z, std::size_t* desc_z,
              int* info)
{
    int* desc_a_ = new int[9];
    int* desc_z_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<std::size_t>(desc_a[i]);
        desc_z_[i] = static_cast<std::size_t>(desc_z[i]);
    }

    int N_ = N;

    double* work;
    double work_q;
    int lwork = -1;

    int* iwork;
    int iwork_size;
    int liwork = -1;

    int one = 1;

    pdsyevd_(&jobz, &uplo, &N_, A, &one, &one, desc_a_, W, Z, &one, &one,
             desc_z_, &work_q, &lwork, &iwork_size, &liwork, info);

    int work_size = int(work_q);
    work_size = 1.5 * work_size;
    auto wptr_ = std::unique_ptr<double[]>{new double[work_size]};

    work = wptr_.get();

    auto iwptr_ = std::unique_ptr<int[]>{new int[iwork_size]};

    iwork = iwptr_.get();

    pdsyevd_(&jobz, &uplo, &N_, A, &one, &one, desc_a_, W, Z, &one, &one,
             desc_z_, work, &work_size, iwork, &iwork_size, info);

    assert(*info == 0);
}

template <>
void t_pheevd(char jobz, char uplo, std::size_t N, std::complex<float>* A,
              std::size_t* desc_a, float* W, std::complex<float>* Z,
              std::size_t* desc_z, int* info)
{
    int* desc_a_ = new int[9];
    int* desc_z_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<std::size_t>(desc_a[i]);
        desc_z_[i] = static_cast<std::size_t>(desc_z[i]);
    }

    int N_ = N;

    std::complex<float>* work;
    std::complex<float> work_q;
    int lwork = -1;

    float* rwork;
    float rwork_q;
    int lrwork = -1;

    int* iwork;
    int iwork_size;
    int liwork = -1;

    int one = 1;

    pcheevd_(&jobz, &uplo, &N_, A, &one, &one, desc_a_, W, Z, &one, &one,
             desc_z_, &work_q, &lwork, &rwork_q, &lrwork, &iwork_size, &liwork,
             info);

    int work_size = 1.5 * (int)real(work_q);

    int rwork_size = int(rwork_q);

    auto wptr_ = std::unique_ptr<std::complex<float>[]>{
        new std::complex<float>[work_size]};

    work = wptr_.get();

    auto rwptr_ = std::unique_ptr<float[]>{new float[rwork_size]};

    rwork = rwptr_.get();

    auto iwptr_ = std::unique_ptr<int[]>{new int[iwork_size]};

    iwork = iwptr_.get();

    pcheevd_(&jobz, &uplo, &N_, A, &one, &one, desc_a_, W, Z, &one, &one,
             desc_z_, work, &work_size, rwork, &rwork_size, iwork, &iwork_size,
             info);

    assert(*info == 0);
}

template <>
void t_pheevd(char jobz, char uplo, std::size_t N, std::complex<double>* A,
              std::size_t* desc_a, double* W, std::complex<double>* Z,
              std::size_t* desc_z, int* info)
{
    int* desc_a_ = new int[9];
    int* desc_z_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<std::size_t>(desc_a[i]);
        desc_z_[i] = static_cast<std::size_t>(desc_z[i]);
    }

    int N_ = N;

    std::complex<double>* work;
    std::complex<double> work_q;
    int lwork = -1;

    double* rwork;
    double rwork_q;
    int lrwork = -1;

    int* iwork;
    int iwork_size;
    int liwork = -1;

    int one = 1;

    pzheevd_(&jobz, &uplo, &N_, A, &one, &one, desc_a_, W, Z, &one, &one,
             desc_z_, &work_q, &lwork, &rwork_q, &lrwork, &iwork_size, &liwork,
             info);

    int work_size = 1.5 * (int)real(work_q);
    int rwork_size = int(rwork_q);

    auto wptr_ = std::unique_ptr<std::complex<double>[]>{
        new std::complex<double>[work_size]};

    work = wptr_.get();

    auto rwptr_ = std::unique_ptr<double[]>{new double[rwork_size]};

    rwork = rwptr_.get();

    auto iwptr_ = std::unique_ptr<int[]>{new int[iwork_size]};

    iwork = iwptr_.get();

    pzheevd_(&jobz, &uplo, &N_, A, &one, &one, desc_a_, W, Z, &one, &one,
             desc_z_, work, &work_size, rwork, &rwork_size, iwork, &iwork_size,
             info);

    assert(*info == 0);
}

template <typename T>
void t_pgemr2d(std::size_t m, std::size_t n, T* A, int ia, int ja,
               std::size_t* desc_a, T* B, int ib, int jb, std::size_t* desc_b,
               int ictxt)
{
    // Convert std::size_t to int for ScaLAPACK
    int im = static_cast<int>(m);
    int in = static_cast<int>(n);
    int* int_desc_a = reinterpret_cast<int*>(desc_a);
    int* int_desc_b = reinterpret_cast<int*>(desc_b);

    if constexpr (std::is_same_v<T, float>)
    {
        psgemr2d_(&im, &in, A, &ia, &ja, int_desc_a, B, &ib, &jb, int_desc_b,
                  &ictxt);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        pdgemr2d_(&im, &in, A, &ia, &ja, int_desc_a, B, &ib, &jb, int_desc_b,
                  &ictxt);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        pcgemr2d_(&im, &in, A, &ia, &ja, int_desc_a, B, &ib, &jb, int_desc_b,
                  &ictxt);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        pzgemr2d_(&im, &in, A, &ia, &ja, int_desc_a, B, &ib, &jb, int_desc_b,
                  &ictxt);
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, std::complex<float>> ||
                          std::is_same_v<T, std::complex<double>>,
                      "Unsupported type for t_pgemr2d");
    }
}

template <>
void t_pgesvd(char jobu, char jobvt, std::size_t m, std::size_t n, double* A,
              int ia, int ja, std::size_t* desc_a, double* s, double* U, int iu,
              int ju, std::size_t* desc_u, double* VT, int ivt, int jvt,
              std::size_t* desc_vt, int* info)
{
    int* desc_a_ = new int[9];
    int* desc_u_ = new int[9];
    int* desc_vt_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<int>(desc_a[i]);
        desc_u_[i] = static_cast<int>(desc_u[i]);
        desc_vt_[i] = static_cast<int>(desc_vt[i]);
    }

    int m_ = m;
    int n_ = n;

    double* work;
    double numwork;
    int lwork, info_;
    lwork = -1;

    pdgesvd_(&jobu, &jobvt, &m_, &n_, A, &ia, &ja, desc_a_, s, U, &iu, &ju,
             desc_u_, VT, &ivt, &jvt, desc_vt_, &numwork, &lwork, &info_);

    lwork = (int)numwork;

    auto wptr = std::unique_ptr<double[]>{new double[lwork]};
    work = wptr.get();

    pdgesvd_(&jobu, &jobvt, &m_, &n_, A, &ia, &ja, desc_a_, s, U, &iu, &ju,
             desc_u_, VT, &ivt, &jvt, desc_vt_, work, &lwork, &info_);

    *info = info_;
    // Note: info != 0 indicates SVD convergence issues or parameter errors
    // Let the caller handle the error instead of asserting
}

template <>
void t_pgesvd(char jobu, char jobvt, std::size_t m, std::size_t n, float* A,
              int ia, int ja, std::size_t* desc_a, float* s, float* U, int iu,
              int ju, std::size_t* desc_u, float* VT, int ivt, int jvt,
              std::size_t* desc_vt, int* info)
{
    int* desc_a_ = new int[9];
    int* desc_u_ = new int[9];
    int* desc_vt_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<int>(desc_a[i]);
        desc_u_[i] = static_cast<int>(desc_u[i]);
        desc_vt_[i] = static_cast<int>(desc_vt[i]);
    }

    int m_ = m;
    int n_ = n;

    float* work;
    float numwork;
    int lwork, info_;
    lwork = -1;

    psgesvd_(&jobu, &jobvt, &m_, &n_, A, &ia, &ja, desc_a_, s, U, &iu, &ju,
             desc_u_, VT, &ivt, &jvt, desc_vt_, &numwork, &lwork, &info_);

    lwork = (int)numwork;

    auto wptr = std::unique_ptr<float[]>{new float[lwork]};
    work = wptr.get();

    psgesvd_(&jobu, &jobvt, &m_, &n_, A, &ia, &ja, desc_a_, s, U, &iu, &ju,
             desc_u_, VT, &ivt, &jvt, desc_vt_, work, &lwork, &info_);

    *info = info_;
    // Note: info != 0 indicates SVD convergence issues or parameter errors
    // Let the caller handle the error instead of asserting
}

template <>
void t_pgesvd(char jobu, char jobvt, std::size_t m, std::size_t n,
              std::complex<double>* A, int ia, int ja, std::size_t* desc_a,
              double* s, std::complex<double>* U, int iu, int ju,
              std::size_t* desc_u, std::complex<double>* VT, int ivt, int jvt,
              std::size_t* desc_vt, int* info)
{
    int* desc_a_ = new int[9];
    int* desc_u_ = new int[9];
    int* desc_vt_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<int>(desc_a[i]);
        desc_u_[i] = static_cast<int>(desc_u[i]);
        desc_vt_[i] = static_cast<int>(desc_vt[i]);
    }

    int m_ = m;
    int n_ = n;

    std::complex<double>* work;
    std::complex<double> numwork;
    int lwork, info_;
    lwork = -1;

    double* rwork;
    int lrwork = 1 + 4 * std::max(m, n);

    auto rwptr = std::unique_ptr<double[]>{new double[lrwork]};
    rwork = rwptr.get();

    pzgesvd_(&jobu, &jobvt, &m_, &n_, A, &ia, &ja, desc_a_, s, U, &iu, &ju,
             desc_u_, VT, &ivt, &jvt, desc_vt_, &numwork, &lwork, rwork,
             &info_);

    lwork = (int)real(numwork);

    auto wptr = std::unique_ptr<std::complex<double>[]>{
        new std::complex<double>[lwork]};
    work = wptr.get();

    pzgesvd_(&jobu, &jobvt, &m_, &n_, A, &ia, &ja, desc_a_, s, U, &iu, &ju,
             desc_u_, VT, &ivt, &jvt, desc_vt_, work, &lwork, rwork, &info_);

    *info = info_;
    // Note: info != 0 indicates SVD convergence issues or parameter errors
    // Let the caller handle the error instead of asserting
}

template <>
void t_pgesvd(char jobu, char jobvt, std::size_t m, std::size_t n,
              std::complex<float>* A, int ia, int ja, std::size_t* desc_a,
              float* s, std::complex<float>* U, int iu, int ju,
              std::size_t* desc_u, std::complex<float>* VT, int ivt, int jvt,
              std::size_t* desc_vt, int* info)
{
    int* desc_a_ = new int[9];
    int* desc_u_ = new int[9];
    int* desc_vt_ = new int[9];

    for (int i = 0; i < 9; i++)
    {
        desc_a_[i] = static_cast<int>(desc_a[i]);
        desc_u_[i] = static_cast<int>(desc_u[i]);
        desc_vt_[i] = static_cast<int>(desc_vt[i]);
    }

    int m_ = m;
    int n_ = n;

    std::complex<float>* work;
    std::complex<float> numwork;
    int lwork, info_;
    lwork = -1;

    float* rwork;
    int lrwork = 1 + 4 * std::max(m, n);

    auto rwptr = std::unique_ptr<float[]>{new float[lrwork]};
    rwork = rwptr.get();

    pcgesvd_(&jobu, &jobvt, &m_, &n_, A, &ia, &ja, desc_a_, s, U, &iu, &ju,
             desc_u_, VT, &ivt, &jvt, desc_vt_, &numwork, &lwork, rwork,
             &info_);

    lwork = (int)real(numwork);

    auto wptr =
        std::unique_ptr<std::complex<float>[]>{new std::complex<float>[lwork]};
    work = wptr.get();

    pcgesvd_(&jobu, &jobvt, &m_, &n_, A, &ia, &ja, desc_a_, s, U, &iu, &ju,
             desc_u_, VT, &ivt, &jvt, desc_vt_, work, &lwork, rwork, &info_);

    *info = info_;
    // Note: info != 0 indicates SVD convergence issues or parameter errors
    // Let the caller handle the error instead of asserting
}

} // namespace scalapackpp
} // namespace linalg
} // namespace chase