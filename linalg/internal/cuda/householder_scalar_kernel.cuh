// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstddef>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include "highprec_traits.cuh"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

//-----------------------------------------------------------------------------
// Fused Householder scalar kernel (single thread): from x0 and nrm_sq compute
// tau, inv_denom, neg_beta, denom_bcast, saved_rkk on device. No D2H except
// for optional tau skip check.
// pivot_here: 1 on pivot rank, 0 elsewhere. When 0, all outputs are set to 0.
// d_x0: on pivot rank = V[pivot_loc + k*ldv]; on others pass pointer to zero.
// d_nrm_sq: global_nrm_sq (one RealT), same on all ranks after Allreduce.
// Outputs: d_tau, d_inv_denom, d_neg_beta, d_denom_bcast, d_saved_rkk (all single element).
//-----------------------------------------------------------------------------

__global__ void householder_scalar_kernel_s(
    int pivot_here, const float* d_x0, const float* d_nrm_sq,
    float* d_tau, float* d_inv_denom, float* d_neg_beta,
    float* d_denom_bcast, float* d_saved_rkk)
{
    if (!pivot_here)
    {
        *d_tau = 0;
        *d_inv_denom = 0;
        *d_neg_beta = 0;
        *d_denom_bcast = 0;
        *d_saved_rkk = 0;
        return;
    }
    float nrm_sq = *d_nrm_sq;
    float x0 = *d_x0;
    if (!(nrm_sq > 0.f) || !isfinite(nrm_sq))
    {
        *d_tau = 0;
        *d_inv_denom = 0;
        *d_neg_beta = 0;
        *d_denom_bcast = 0;
        *d_saved_rkk = x0;
        return;
    }
    float nrm = sqrtf(nrm_sq);
    if (!isfinite(nrm))
    {
        *d_tau = 0;
        *d_inv_denom = 0;
        *d_neg_beta = 0;
        *d_denom_bcast = 0;
        *d_saved_rkk = x0;
        return;
    }
    float beta = (x0 >= 0.f) ? nrm : (-nrm);
    float denom = x0 + beta;
    if (denom == 0.f || !isfinite(denom) || !isfinite(beta))
    {
        *d_tau = 0;
        *d_inv_denom = 0;
        *d_neg_beta = 0;
        *d_denom_bcast = 0;
        *d_saved_rkk = x0;
        return;
    }
    *d_tau = denom / beta;
    *d_inv_denom = 1.f / denom;
    if (!isfinite(*d_tau) || !isfinite(*d_inv_denom))
    {
        *d_tau = 0;
        *d_inv_denom = 0;
        *d_neg_beta = 0;
        *d_denom_bcast = 0;
        *d_saved_rkk = x0;
        return;
    }
    *d_neg_beta = -beta;
    *d_denom_bcast = denom;  // = tau*beta, same as reference
    *d_saved_rkk = x0;
}

__global__ void householder_scalar_kernel_d(
    int pivot_here, const double* d_x0, const double* d_nrm_sq,
    double* d_tau, double* d_inv_denom, double* d_neg_beta,
    double* d_denom_bcast, double* d_saved_rkk)
{
    if (!pivot_here)
    {
        *d_tau = 0;
        *d_inv_denom = 0;
        *d_neg_beta = 0;
        *d_denom_bcast = 0;
        *d_saved_rkk = 0;
        return;
    }
    double nrm_sq = *d_nrm_sq;
    double x0 = *d_x0;
    if (!(nrm_sq > 0.) || !isfinite(nrm_sq))
    {
        *d_tau = 0;
        *d_inv_denom = 0;
        *d_neg_beta = 0;
        *d_denom_bcast = 0;
        *d_saved_rkk = x0;
        return;
    }
    double nrm = sqrt(nrm_sq);
    if (!isfinite(nrm))
    {
        *d_tau = 0;
        *d_inv_denom = 0;
        *d_neg_beta = 0;
        *d_denom_bcast = 0;
        *d_saved_rkk = x0;
        return;
    }
    double beta = (x0 >= 0.) ? nrm : (-nrm);
    double denom = x0 + beta;
    if (denom == 0. || !isfinite(denom) || !isfinite(beta))
    {
        *d_tau = 0;
        *d_inv_denom = 0;
        *d_neg_beta = 0;
        *d_denom_bcast = 0;
        *d_saved_rkk = x0;
        return;
    }
    *d_tau = denom / beta;
    *d_inv_denom = 1. / denom;
    if (!isfinite(*d_tau) || !isfinite(*d_inv_denom))
    {
        *d_tau = 0;
        *d_inv_denom = 0;
        *d_neg_beta = 0;
        *d_denom_bcast = 0;
        *d_saved_rkk = x0;
        return;
    }
    *d_neg_beta = -beta;
    *d_denom_bcast = denom;  // = tau*beta, same as reference
    *d_saved_rkk = x0;
}

__device__ static cuComplex cuCdivf_safe(cuComplex a, float b)
{
    if (b == 0.f)
        return make_cuComplex(0.f, 0.f);
    return make_cuComplex(cuCrealf(a) / b, cuCimagf(a) / b);
}

__global__ void householder_scalar_kernel_c(
    int pivot_here, const cuComplex* d_x0, const float* d_nrm_sq,
    cuComplex* d_tau, cuComplex* d_inv_denom, cuComplex* d_neg_beta,
    cuComplex* d_denom_bcast, cuComplex* d_saved_rkk)
{
    if (!pivot_here)
    {
        *d_tau = make_cuComplex(0.f, 0.f);
        *d_inv_denom = make_cuComplex(0.f, 0.f);
        *d_neg_beta = make_cuComplex(0.f, 0.f);
        *d_denom_bcast = make_cuComplex(0.f, 0.f);
        *d_saved_rkk = make_cuComplex(0.f, 0.f);
        return;
    }
    float nrm_sq = *d_nrm_sq;
    cuComplex x0 = *d_x0;
    if (!(nrm_sq > 0.f) || !isfinite(nrm_sq))
    {
        *d_tau = make_cuComplex(0.f, 0.f);
        *d_inv_denom = make_cuComplex(0.f, 0.f);
        *d_neg_beta = make_cuComplex(0.f, 0.f);
        *d_denom_bcast = make_cuComplex(0.f, 0.f);
        *d_saved_rkk = x0;
        return;
    }
    float nrm = sqrtf(nrm_sq);
    if (!isfinite(nrm))
    {
        *d_tau = make_cuComplex(0.f, 0.f);
        *d_inv_denom = make_cuComplex(0.f, 0.f);
        *d_neg_beta = make_cuComplex(0.f, 0.f);
        *d_denom_bcast = make_cuComplex(0.f, 0.f);
        *d_saved_rkk = x0;
        return;
    }
    float ax0 = cuCabsf(x0);
    if (!isfinite(ax0))
    {
        *d_tau = make_cuComplex(0.f, 0.f);
        *d_inv_denom = make_cuComplex(0.f, 0.f);
        *d_neg_beta = make_cuComplex(0.f, 0.f);
        *d_denom_bcast = make_cuComplex(0.f, 0.f);
        *d_saved_rkk = x0;
        return;
    }
    cuComplex sign_x0 = (ax0 == 0.f) ? make_cuComplex(1.f, 0.f) : cuCdivf_safe(x0, ax0);
    cuComplex beta = make_cuComplex(cuCrealf(sign_x0) * nrm, cuCimagf(sign_x0) * nrm);
    cuComplex denom = cuCaddf(x0, beta);
    float adenom = cuCabsf(denom);
    if (adenom == 0.f || !isfinite(adenom))
    {
        *d_tau = make_cuComplex(0.f, 0.f);
        *d_inv_denom = make_cuComplex(0.f, 0.f);
        *d_neg_beta = make_cuComplex(-cuCrealf(beta), -cuCimagf(beta));
        *d_denom_bcast = make_cuComplex(0.f, 0.f);
        *d_saved_rkk = x0;
        return;
    }
    *d_tau = cuCdivf(denom, beta);
    *d_inv_denom = cuCdivf(make_cuComplex(1.f, 0.f), denom);
    if (!isfinite(cuCrealf(*d_tau)) || !isfinite(cuCimagf(*d_tau)) ||
        !isfinite(cuCrealf(*d_inv_denom)) || !isfinite(cuCimagf(*d_inv_denom)))
    {
        *d_tau = make_cuComplex(0.f, 0.f);
        *d_inv_denom = make_cuComplex(0.f, 0.f);
        *d_neg_beta = make_cuComplex(0.f, 0.f);
        *d_denom_bcast = make_cuComplex(0.f, 0.f);
        *d_saved_rkk = x0;
        return;
    }
    *d_neg_beta = make_cuComplex(-cuCrealf(beta), -cuCimagf(beta));
    *d_denom_bcast = denom;  // = tau*beta, same as reference
    *d_saved_rkk = x0;
}

__device__ static cuDoubleComplex cuCdiv_safe(cuDoubleComplex a, double b)
{
    if (b == 0.)
        return make_cuDoubleComplex(0., 0.);
    return make_cuDoubleComplex(cuCreal(a) / b, cuCimag(a) / b);
}

__global__ void householder_scalar_kernel_z(
    int pivot_here, const cuDoubleComplex* d_x0, const double* d_nrm_sq,
    cuDoubleComplex* d_tau, cuDoubleComplex* d_inv_denom, cuDoubleComplex* d_neg_beta,
    cuDoubleComplex* d_denom_bcast, cuDoubleComplex* d_saved_rkk)
{
    if (!pivot_here)
    {
        *d_tau = make_cuDoubleComplex(0., 0.);
        *d_inv_denom = make_cuDoubleComplex(0., 0.);
        *d_neg_beta = make_cuDoubleComplex(0., 0.);
        *d_denom_bcast = make_cuDoubleComplex(0., 0.);
        *d_saved_rkk = make_cuDoubleComplex(0., 0.);
        return;
    }
    double nrm_sq = *d_nrm_sq;
    cuDoubleComplex x0 = *d_x0;
    if (!(nrm_sq > 0.) || !isfinite(nrm_sq))
    {
        *d_tau = make_cuDoubleComplex(0., 0.);
        *d_inv_denom = make_cuDoubleComplex(0., 0.);
        *d_neg_beta = make_cuDoubleComplex(0., 0.);
        *d_denom_bcast = make_cuDoubleComplex(0., 0.);
        *d_saved_rkk = x0;
        return;
    }
    double nrm = sqrt(nrm_sq);
    if (!isfinite(nrm))
    {
        *d_tau = make_cuDoubleComplex(0., 0.);
        *d_inv_denom = make_cuDoubleComplex(0., 0.);
        *d_neg_beta = make_cuDoubleComplex(0., 0.);
        *d_denom_bcast = make_cuDoubleComplex(0., 0.);
        *d_saved_rkk = x0;
        return;
    }
    double ax0 = cuCabs(x0);
    if (!isfinite(ax0))
    {
        *d_tau = make_cuDoubleComplex(0., 0.);
        *d_inv_denom = make_cuDoubleComplex(0., 0.);
        *d_neg_beta = make_cuDoubleComplex(0., 0.);
        *d_denom_bcast = make_cuDoubleComplex(0., 0.);
        *d_saved_rkk = x0;
        return;
    }
    cuDoubleComplex sign_x0 = (ax0 == 0.) ? make_cuDoubleComplex(1., 0.) : cuCdiv_safe(x0, ax0);
    cuDoubleComplex beta = make_cuDoubleComplex(cuCreal(sign_x0) * nrm, cuCimag(sign_x0) * nrm);
    cuDoubleComplex denom = cuCadd(x0, beta);
    double adenom = cuCabs(denom);
    if (adenom == 0. || !isfinite(adenom))
    {
        *d_tau = make_cuDoubleComplex(0., 0.);
        *d_inv_denom = make_cuDoubleComplex(0., 0.);
        *d_neg_beta = make_cuDoubleComplex(-cuCreal(beta), -cuCimag(beta));
        *d_denom_bcast = make_cuDoubleComplex(0., 0.);
        *d_saved_rkk = x0;
        return;
    }
    *d_tau = cuCdiv(denom, beta);
    *d_inv_denom = cuCdiv(make_cuDoubleComplex(1., 0.), denom);
    if (!isfinite(cuCreal(*d_tau)) || !isfinite(cuCimag(*d_tau)) ||
        !isfinite(cuCreal(*d_inv_denom)) || !isfinite(cuCimag(*d_inv_denom)))
    {
        *d_tau = make_cuDoubleComplex(0., 0.);
        *d_inv_denom = make_cuDoubleComplex(0., 0.);
        *d_neg_beta = make_cuDoubleComplex(0., 0.);
        *d_denom_bcast = make_cuDoubleComplex(0., 0.);
        *d_saved_rkk = x0;
        return;
    }
    *d_neg_beta = make_cuDoubleComplex(-cuCreal(beta), -cuCimag(beta));
    *d_denom_bcast = denom;  // = tau*beta, same as reference
    *d_saved_rkk = x0;
}

//-----------------------------------------------------------------------------
// After Allreduce(denom_bcast): set d_inv_denom = 1/d_denom_bcast when
// d_denom_bcast != 0, else 0. Single element, run on all ranks.
//-----------------------------------------------------------------------------

__global__ void inv_denom_from_denom_bcast_s(
    const float* d_denom_bcast, float* d_inv_denom)
{
    float x = *d_denom_bcast;
    *d_inv_denom = (x != 0.f && isfinite(x)) ? (1.f / x) : 0.f;
}

__global__ void inv_denom_from_denom_bcast_d(
    const double* d_denom_bcast, double* d_inv_denom)
{
    double x = *d_denom_bcast;
    *d_inv_denom = (x != 0. && isfinite(x)) ? (1. / x) : 0.;
}

__global__ void inv_denom_from_denom_bcast_c(
    const cuComplex* d_denom_bcast, cuComplex* d_inv_denom)
{
    cuComplex x = *d_denom_bcast;
    float ax = cuCabsf(x);
    *d_inv_denom = (ax != 0.f && isfinite(ax))
        ? cuCdivf(make_cuComplex(1.f, 0.f), x) : make_cuComplex(0.f, 0.f);
}

__global__ void inv_denom_from_denom_bcast_z(
    const cuDoubleComplex* d_denom_bcast, cuDoubleComplex* d_inv_denom)
{
    cuDoubleComplex x = *d_denom_bcast;
    double ax = cuCabs(x);
    *d_inv_denom = (ax != 0. && isfinite(ax))
        ? cuCdiv(make_cuDoubleComplex(1., 0.), x) : make_cuDoubleComplex(0., 0.);
}

//-----------------------------------------------------------------------------
// Fused: if denom_bcast != 0 then set inv_denom = 1/denom_bcast and scale
// V_col[0..n-1] by inv_denom. Avoids D2H for denom_bcast branch in panel.
//-----------------------------------------------------------------------------
__global__ void nonpivot_scal_if_denom_nonzero_s(
    const float* d_denom_bcast, float* d_inv_denom,
    float* d_V_col, int n)
{
    float x = *d_denom_bcast;
    if (x == 0.f)
        return;
    float inv = 1.f / x;
    *d_inv_denom = inv;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
        d_V_col[i] *= inv;
}

__global__ void nonpivot_scal_if_denom_nonzero_d(
    const double* d_denom_bcast, double* d_inv_denom,
    double* d_V_col, int n)
{
    double x = *d_denom_bcast;
    if (x == 0.)
        return;
    double inv = 1. / x;
    *d_inv_denom = inv;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
        d_V_col[i] *= inv;
}

__global__ void nonpivot_scal_if_denom_nonzero_c(
    const cuComplex* d_denom_bcast, cuComplex* d_inv_denom,
    cuComplex* d_V_col, int n)
{
    cuComplex x = *d_denom_bcast;
    if (cuCabsf(x) == 0.f)
        return;
    cuComplex inv = cuCdivf(make_cuComplex(1.f, 0.f), x);
    *d_inv_denom = inv;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
        d_V_col[i] = cuCmulf(d_V_col[i], inv);
}

__global__ void nonpivot_scal_if_denom_nonzero_z(
    const cuDoubleComplex* d_denom_bcast, cuDoubleComplex* d_inv_denom,
    cuDoubleComplex* d_V_col, int n)
{
    cuDoubleComplex x = *d_denom_bcast;
    if (cuCabs(x) == 0.)
        return;
    cuDoubleComplex inv = cuCdiv(make_cuDoubleComplex(1., 0.), x);
    *d_inv_denom = inv;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
        d_V_col[i] = cuCmul(d_V_col[i], inv);
}

//-----------------------------------------------------------------------------
// Block-cyclic panel: after NCCL(tau, denom), multi-block grid-stride SCAL +
// pivot row := 1 without a second launch (pivot index pivot_rel skips *= inv).
// pivot_rel: offset in d_v_tail for pivot row; -1 if !pivot_here.
//-----------------------------------------------------------------------------
__global__ void bc1d_post_comm_scal_pivot_s(
    int pivot_here, int pivot_rel, const float* d_denom_bcast, float* d_inv_denom,
    float* d_v_tail, int vr)
{
    float x = *d_denom_bcast;
    const bool do_scale = (x != 0.f && isfinite(x));
    float inv = 0.f;
    if (do_scale)
        inv = 1.f / x;
    if (blockIdx.x == 0 && threadIdx.x == 0 && do_scale)
        *d_inv_denom = inv;

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < vr; i += stride)
    {
        const bool is_piv =
            (pivot_here != 0 && pivot_rel >= 0 && i == pivot_rel);
        if (is_piv)
            d_v_tail[i] = 1.f;
        else if (do_scale)
            d_v_tail[i] *= inv;
    }
}

__global__ void bc1d_post_comm_scal_pivot_d(
    int pivot_here, int pivot_rel, const double* d_denom_bcast, double* d_inv_denom,
    double* d_v_tail, int vr)
{
    double x = *d_denom_bcast;
    const bool do_scale = (x != 0. && isfinite(x));
    double inv = 0.;
    if (do_scale)
        inv = 1. / x;
    if (blockIdx.x == 0 && threadIdx.x == 0 && do_scale)
        *d_inv_denom = inv;

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < vr; i += stride)
    {
        const bool is_piv =
            (pivot_here != 0 && pivot_rel >= 0 && i == pivot_rel);
        if (is_piv)
            d_v_tail[i] = 1.;
        else if (do_scale)
            d_v_tail[i] *= inv;
    }
}

__global__ void bc1d_post_comm_scal_pivot_c(
    int pivot_here, int pivot_rel, const cuComplex* d_denom_bcast, cuComplex* d_inv_denom,
    cuComplex* d_v_tail, int vr)
{
    cuComplex x = *d_denom_bcast;
    const float ax = cuCabsf(x);
    const bool do_scale = (ax != 0.f && isfinite(ax));
    cuComplex inv = make_cuComplex(0.f, 0.f);
    if (do_scale)
        inv = cuCdivf(make_cuComplex(1.f, 0.f), x);
    if (blockIdx.x == 0 && threadIdx.x == 0 && do_scale)
        *d_inv_denom = inv;

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < vr; i += stride)
    {
        const bool is_piv =
            (pivot_here != 0 && pivot_rel >= 0 && i == pivot_rel);
        if (is_piv)
            d_v_tail[i] = make_cuComplex(1.f, 0.f);
        else if (do_scale)
            d_v_tail[i] = cuCmulf(d_v_tail[i], inv);
    }
}

__global__ void bc1d_post_comm_scal_pivot_z(
    int pivot_here, int pivot_rel, const cuDoubleComplex* d_denom_bcast,
    cuDoubleComplex* d_inv_denom, cuDoubleComplex* d_v_tail, int vr)
{
    cuDoubleComplex x = *d_denom_bcast;
    const double ax = cuCabs(x);
    const bool do_scale = (ax != 0. && isfinite(ax));
    cuDoubleComplex inv = make_cuDoubleComplex(0., 0.);
    if (do_scale)
        inv = cuCdiv(make_cuDoubleComplex(1., 0.), x);
    if (blockIdx.x == 0 && threadIdx.x == 0 && do_scale)
        *d_inv_denom = inv;

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < vr; i += stride)
    {
        const bool is_piv =
            (pivot_here != 0 && pivot_rel >= 0 && i == pivot_rel);
        if (is_piv)
            d_v_tail[i] = make_cuDoubleComplex(1., 0.);
        else if (do_scale)
            d_v_tail[i] = cuCmul(d_v_tail[i], inv);
    }
}

//-----------------------------------------------------------------------------
// Guarded scaling for panel factorization: if tau == 0, no-op; otherwise scale
// by inv_denom and set pivot entry to neg_beta on pivot rank.
//-----------------------------------------------------------------------------
__global__ void guarded_scaling_s(
    int n, const float* d_tau, const float* d_inv_denom, float* d_V_col,
    int pivot_here, int pivot_loc, const float* d_neg_beta)
{
    if (*d_tau == 0.f || !isfinite(*d_tau) || !isfinite(*d_inv_denom))
        return;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    if (pivot_here && idx == pivot_loc)
        d_V_col[idx] = *d_neg_beta;
    else
        d_V_col[idx] *= *d_inv_denom;
}

__global__ void guarded_scaling_d(
    int n, const double* d_tau, const double* d_inv_denom, double* d_V_col,
    int pivot_here, int pivot_loc, const double* d_neg_beta)
{
    if (*d_tau == 0. || !isfinite(*d_tau) || !isfinite(*d_inv_denom))
        return;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    if (pivot_here && idx == pivot_loc)
        d_V_col[idx] = *d_neg_beta;
    else
        d_V_col[idx] *= *d_inv_denom;
}

__global__ void guarded_scaling_c(
    int n, const cuComplex* d_tau, const cuComplex* d_inv_denom, cuComplex* d_V_col,
    int pivot_here, int pivot_loc, const cuComplex* d_neg_beta)
{
    if (cuCabsf(*d_tau) == 0.f || !isfinite(cuCabsf(*d_tau)) ||
        !isfinite(cuCrealf(*d_inv_denom)) || !isfinite(cuCimagf(*d_inv_denom)))
        return;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    if (pivot_here && idx == pivot_loc)
        d_V_col[idx] = *d_neg_beta;
    else
        d_V_col[idx] = cuCmulf(d_V_col[idx], *d_inv_denom);
}

__global__ void guarded_scaling_z(
    int n, const cuDoubleComplex* d_tau, const cuDoubleComplex* d_inv_denom, cuDoubleComplex* d_V_col,
    int pivot_here, int pivot_loc, const cuDoubleComplex* d_neg_beta)
{
    if (cuCabs(*d_tau) == 0. || !isfinite(cuCabs(*d_tau)) ||
        !isfinite(cuCreal(*d_inv_denom)) || !isfinite(cuCimag(*d_inv_denom)))
        return;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    if (pivot_here && idx == pivot_loc)
        d_V_col[idx] = *d_neg_beta;
    else
        d_V_col[idx] = cuCmul(d_V_col[idx], *d_inv_denom);
}

//-----------------------------------------------------------------------------
// Batch save/restore of upper-triangular panel entries.
// Uses packed upper-triangular indexing to avoid O(jb^2) host memcpy calls.
// Keep typed kernels (including cuComplex/cuDoubleComplex) so complex handling
// is explicit and mirrors memcpy-style value movement.
//-----------------------------------------------------------------------------
__global__ void init_identity_distributed_s(float* d_V, std::size_t ldv,
                                            std::size_t n, std::size_t g_off,
                                            std::size_t l_rows)
{
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n)
        return;

    float* d_col = d_V + col * ldv;
    for (std::size_t i = 0; i < l_rows; ++i)
        d_col[i] = 0.0f;

    if (col >= g_off && col < g_off + l_rows)
        d_col[col - g_off] = 1.0f;
}

__global__ void init_identity_distributed_d(double* d_V, std::size_t ldv,
                                            std::size_t n, std::size_t g_off,
                                            std::size_t l_rows)
{
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n)
        return;

    double* d_col = d_V + col * ldv;
    for (std::size_t i = 0; i < l_rows; ++i)
        d_col[i] = 0.0;

    if (col >= g_off && col < g_off + l_rows)
        d_col[col - g_off] = 1.0;
}

__global__ void init_identity_distributed_c(cuComplex* d_V, std::size_t ldv,
                                            std::size_t n, std::size_t g_off,
                                            std::size_t l_rows)
{
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n)
        return;

    cuComplex* d_col = d_V + col * ldv;
    for (std::size_t i = 0; i < l_rows; ++i)
        d_col[i] = make_cuComplex(0.0f, 0.0f);

    if (col >= g_off && col < g_off + l_rows)
        d_col[col - g_off] = make_cuComplex(1.0f, 0.0f);
}

__global__ void init_identity_distributed_z(cuDoubleComplex* d_V, std::size_t ldv,
                                            std::size_t n, std::size_t g_off,
                                            std::size_t l_rows)
{
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= n)
        return;

    cuDoubleComplex* d_col = d_V + col * ldv;
    for (std::size_t i = 0; i < l_rows; ++i)
        d_col[i] = make_cuDoubleComplex(0.0, 0.0);

    if (col >= g_off && col < g_off + l_rows)
        d_col[col - g_off] = make_cuDoubleComplex(1.0, 0.0);
}

__global__ void batch_save_restore_upper_triangular_s(
    float* d_V, float* d_saved, std::size_t ldv, std::size_t jb, std::size_t k,
    std::size_t g_off, std::size_t l_rows,
    int save_mode)
{
    const std::size_t jj =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t ii =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (jj >= jb || ii > jj)
        return;

    const std::size_t grow = k + ii;
    if (grow < g_off || grow >= g_off + l_rows)
        return;

    const std::size_t col = k + jj;
    const std::size_t lrow = grow - g_off;
    float* d_elem = d_V + lrow + col * ldv;
    float* d_slot = d_saved + ii + jj * jb;
    if (save_mode)
    {
        *d_slot = *d_elem;
        *d_elem = (ii == jj) ? 1.0f : 0.0f;
    }
    else
    {
        *d_elem = *d_slot;
    }
}

__global__ void batch_save_restore_upper_triangular_d(
    double* d_V, double* d_saved, std::size_t ldv, std::size_t jb, std::size_t k,
    std::size_t g_off, std::size_t l_rows,
    int save_mode)
{
    const std::size_t jj =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t ii =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (jj >= jb || ii > jj)
        return;

    const std::size_t grow = k + ii;
    if (grow < g_off || grow >= g_off + l_rows)
        return;

    const std::size_t col = k + jj;
    const std::size_t lrow = grow - g_off;
    double* d_elem = d_V + lrow + col * ldv;
    double* d_slot = d_saved + ii + jj * jb;
    if (save_mode)
    {
        *d_slot = *d_elem;
        *d_elem = (ii == jj) ? 1.0 : 0.0;
    }
    else
    {
        *d_elem = *d_slot;
    }
}

__global__ void batch_save_restore_upper_triangular_c(
    cuComplex* d_V, cuComplex* d_saved, std::size_t ldv, std::size_t jb, std::size_t k,
    std::size_t g_off, std::size_t l_rows,
    int save_mode)
{
    const std::size_t jj =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t ii =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (jj >= jb || ii > jj)
        return;

    const std::size_t grow = k + ii;
    if (grow < g_off || grow >= g_off + l_rows)
        return;

    const std::size_t col = k + jj;
    const std::size_t lrow = grow - g_off;
    cuComplex* d_elem = d_V + lrow + col * ldv;
    cuComplex* d_slot = d_saved + ii + jj * jb;
    if (save_mode)
    {
        *d_slot = *d_elem;
        *d_elem = (ii == jj) ? make_cuComplex(1.0f, 0.0f)
                             : make_cuComplex(0.0f, 0.0f);
    }
    else
    {
        *d_elem = *d_slot;
    }
}

__global__ void batch_save_restore_upper_triangular_z(
    cuDoubleComplex* d_V, cuDoubleComplex* d_saved, std::size_t ldv, std::size_t jb, std::size_t k,
    std::size_t g_off, std::size_t l_rows,
    int save_mode)
{
    const std::size_t jj =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t ii =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (jj >= jb || ii > jj)
        return;

    const std::size_t grow = k + ii;
    if (grow < g_off || grow >= g_off + l_rows)
        return;

    const std::size_t col = k + jj;
    const std::size_t lrow = grow - g_off;
    cuDoubleComplex* d_elem = d_V + lrow + col * ldv;
    cuDoubleComplex* d_slot = d_saved + ii + jj * jb;
    if (save_mode)
    {
        *d_slot = *d_elem;
        *d_elem = (ii == jj) ? make_cuDoubleComplex(1.0, 0.0)
                             : make_cuDoubleComplex(0.0, 0.0);
    }
    else
    {
        *d_elem = *d_slot;
    }
}

//-----------------------------------------------------------------------------
// Split-and-Pad: after Householder scaling and intra-panel trailing update,
// enforce WY-clean column v — global row < pivot -> 0; pivot -> 1 with R_kk
// peeled to d_r_diag_out (optional). global row > pivot unchanged.
// d_V_col0 points at V(0, col); entry i is at d_V_col0[i] (column-major, lda rows).
//-----------------------------------------------------------------------------
__global__ void split_and_pad_v_column_s(
    float* d_V_col0, const unsigned long long* d_grow, int l_rows,
    unsigned long long pivot_global, int pivot_here, int pivot_loc,
    const float* d_saved_rkk, float* d_r_diag_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows)
        return;
    const unsigned long long g = d_grow[i];
    float* v = d_V_col0 + i;
    if (g < pivot_global)
    {
        *v = 0.f;
        return;
    }
    if (g == pivot_global && pivot_here && i == pivot_loc)
    {
        if (d_r_diag_out != nullptr)
            *d_r_diag_out = *d_saved_rkk;
        *v = 1.f;
        return;
    }
}

__global__ void split_and_pad_v_column_d(
    double* d_V_col0, const unsigned long long* d_grow, int l_rows,
    unsigned long long pivot_global, int pivot_here, int pivot_loc,
    const double* d_saved_rkk, double* d_r_diag_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows)
        return;
    const unsigned long long g = d_grow[i];
    double* v = d_V_col0 + i;
    if (g < pivot_global)
    {
        *v = 0.;
        return;
    }
    if (g == pivot_global && pivot_here && i == pivot_loc)
    {
        if (d_r_diag_out != nullptr)
            *d_r_diag_out = *d_saved_rkk;
        *v = 1.;
        return;
    }
}

__global__ void split_and_pad_v_column_c(
    cuComplex* d_V_col0, const unsigned long long* d_grow, int l_rows,
    unsigned long long pivot_global, int pivot_here, int pivot_loc,
    const cuComplex* d_saved_rkk, cuComplex* d_r_diag_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows)
        return;
    const unsigned long long g = d_grow[i];
    cuComplex* v = d_V_col0 + i;
    if (g < pivot_global)
    {
        *v = make_cuComplex(0.f, 0.f);
        return;
    }
    if (g == pivot_global && pivot_here && i == pivot_loc)
    {
        if (d_r_diag_out != nullptr)
            *d_r_diag_out = *d_saved_rkk;
        *v = make_cuComplex(1.f, 0.f);
        return;
    }
}

__global__ void split_and_pad_v_column_z(
    cuDoubleComplex* d_V_col0, const unsigned long long* d_grow, int l_rows,
    unsigned long long pivot_global, int pivot_here, int pivot_loc,
    const cuDoubleComplex* d_saved_rkk, cuDoubleComplex* d_r_diag_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows)
        return;
    const unsigned long long g = d_grow[i];
    cuDoubleComplex* v = d_V_col0 + i;
    if (g < pivot_global)
    {
        *v = make_cuDoubleComplex(0., 0.);
        return;
    }
    if (g == pivot_global && pivot_here && i == pivot_loc)
    {
        if (d_r_diag_out != nullptr)
            *d_r_diag_out = *d_saved_rkk;
        *v = make_cuDoubleComplex(1., 0.);
        return;
    }
}

//-----------------------------------------------------------------------------
// Stage-3 fused finish for block-cyclic panel column:
// 1) use globally synchronized denom to scale active tail [active_row_start, l_rows)
// 2) split-and-pad cleanup by global row index (g < pivot -> 0)
// 3) pivot row fix-up v[pivot]=1 and optional R_kk peel
//-----------------------------------------------------------------------------
__global__ void fused_householder_finish_s(
    float* d_V_col0, int l_rows, int active_row_start,
    const unsigned long long* d_grow, unsigned long long pivot_global,
    int pivot_here, int pivot_loc, const float* d_denom_bcast,
    float* d_inv_denom, const float* d_saved_rkk, float* d_r_diag_out)
{
    const float x = *d_denom_bcast;
    const bool do_scale = (x != 0.f && isfinite(x));
    const float inv = do_scale ? (1.f / x) : 0.f;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *d_inv_denom = inv;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows) return;
    float* v = d_V_col0 + i;
    const unsigned long long g = d_grow[i];
    if (g < pivot_global) { *v = 0.f; return; }
    if (pivot_here && i == pivot_loc && g == pivot_global)
    {
        if (d_r_diag_out) *d_r_diag_out = *d_saved_rkk;
        *v = 1.f;
        return;
    }
    if (do_scale && i >= active_row_start) *v *= inv;
}

__global__ void fused_householder_finish_d(
    double* d_V_col0, int l_rows, int active_row_start,
    const unsigned long long* d_grow, unsigned long long pivot_global,
    int pivot_here, int pivot_loc, const double* d_denom_bcast,
    double* d_inv_denom, const double* d_saved_rkk, double* d_r_diag_out)
{
    const double x = *d_denom_bcast;
    const bool do_scale = (x != 0. && isfinite(x));
    const double inv = do_scale ? (1. / x) : 0.;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *d_inv_denom = inv;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows) return;
    double* v = d_V_col0 + i;
    const unsigned long long g = d_grow[i];
    if (g < pivot_global) { *v = 0.; return; }
    if (pivot_here && i == pivot_loc && g == pivot_global)
    {
        if (d_r_diag_out) *d_r_diag_out = *d_saved_rkk;
        *v = 1.;
        return;
    }
    if (do_scale && i >= active_row_start) *v *= inv;
}

__global__ void fused_householder_finish_c(
    cuComplex* d_V_col0, int l_rows, int active_row_start,
    const unsigned long long* d_grow, unsigned long long pivot_global,
    int pivot_here, int pivot_loc, const cuComplex* d_denom_bcast,
    cuComplex* d_inv_denom, const cuComplex* d_saved_rkk, cuComplex* d_r_diag_out)
{
    const cuComplex x = *d_denom_bcast;
    const float ax = cuCabsf(x);
    const bool do_scale = (ax != 0.f && isfinite(ax));
    const cuComplex inv = do_scale ? cuCdivf(make_cuComplex(1.f, 0.f), x)
                                   : make_cuComplex(0.f, 0.f);
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *d_inv_denom = inv;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows) return;
    cuComplex* v = d_V_col0 + i;
    const unsigned long long g = d_grow[i];
    if (g < pivot_global) { *v = make_cuComplex(0.f, 0.f); return; }
    if (pivot_here && i == pivot_loc && g == pivot_global)
    {
        if (d_r_diag_out) *d_r_diag_out = *d_saved_rkk;
        *v = make_cuComplex(1.f, 0.f);
        return;
    }
    if (do_scale && i >= active_row_start) *v = cuCmulf(*v, inv);
}

__global__ void fused_householder_finish_z(
    cuDoubleComplex* d_V_col0, int l_rows, int active_row_start,
    const unsigned long long* d_grow, unsigned long long pivot_global,
    int pivot_here, int pivot_loc, const cuDoubleComplex* d_denom_bcast,
    cuDoubleComplex* d_inv_denom, const cuDoubleComplex* d_saved_rkk,
    cuDoubleComplex* d_r_diag_out)
{
    const cuDoubleComplex x = *d_denom_bcast;
    const double ax = cuCabs(x);
    const bool do_scale = (ax != 0. && isfinite(ax));
    const cuDoubleComplex inv = do_scale ? cuCdiv(make_cuDoubleComplex(1., 0.), x)
                                         : make_cuDoubleComplex(0., 0.);
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *d_inv_denom = inv;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows) return;
    cuDoubleComplex* v = d_V_col0 + i;
    const unsigned long long g = d_grow[i];
    if (g < pivot_global) { *v = make_cuDoubleComplex(0., 0.); return; }
    if (pivot_here && i == pivot_loc && g == pivot_global)
    {
        if (d_r_diag_out) *d_r_diag_out = *d_saved_rkk;
        *v = make_cuDoubleComplex(1., 0.);
        return;
    }
    if (do_scale && i >= active_row_start) *v = cuCmul(*v, inv);
}

__global__ void extract_real_part_from_c(const cuComplex* in, float* out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = cuCrealf(*in);
}

__global__ void extract_real_part_from_z(const cuDoubleComplex* in, double* out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = cuCreal(*in);
}

__global__ void copy_scalar_s(const float* src, float* dst)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *dst = *src;
}
__global__ void copy_scalar_d(const double* src, double* dst)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *dst = *src;
}
__global__ void copy_scalar_c(const cuComplex* src, cuComplex* dst)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *dst = *src;
}
__global__ void copy_scalar_z(const cuDoubleComplex* src, cuDoubleComplex* dst)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *dst = *src;
}

//-----------------------------------------------------------------------------
// Panel pre-clean: V columns [k, k+jb) — zero entries on local rows with
// global index < k_col0 only (no pivot patch, no R_kk peel). One launch for
// the whole panel strip; avoids wiping valid upper-triangle data on future pivots.
//-----------------------------------------------------------------------------
__global__ void panel_pre_clean_s(float* d_V_panel, int ldv,
    const unsigned long long* d_grow, int l_rows,
    unsigned long long k_col0, int jb_cols)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows || jb_cols <= 0)
        return;
    if (d_grow[i] >= k_col0)
        return;
    for (int j = 0; j < jb_cols; ++j)
        d_V_panel[i + j * ldv] = 0.f;
}

__global__ void panel_pre_clean_d(double* d_V_panel, int ldv,
    const unsigned long long* d_grow, int l_rows,
    unsigned long long k_col0, int jb_cols)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows || jb_cols <= 0)
        return;
    if (d_grow[i] >= k_col0)
        return;
    for (int j = 0; j < jb_cols; ++j)
        d_V_panel[i + j * ldv] = 0.;
}

__global__ void panel_pre_clean_c(cuComplex* d_V_panel, int ldv,
    const unsigned long long* d_grow, int l_rows,
    unsigned long long k_col0, int jb_cols)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows || jb_cols <= 0)
        return;
    if (d_grow[i] >= k_col0)
        return;
    const cuComplex z0 = make_cuComplex(0.f, 0.f);
    for (int j = 0; j < jb_cols; ++j)
        d_V_panel[i + j * ldv] = z0;
}

__global__ void panel_pre_clean_z(cuDoubleComplex* d_V_panel, int ldv,
    const unsigned long long* d_grow, int l_rows,
    unsigned long long k_col0, int jb_cols)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= l_rows || jb_cols <= 0)
        return;
    if (d_grow[i] >= k_col0)
        return;
    const cuDoubleComplex z0 = make_cuDoubleComplex(0., 0.);
    for (int j = 0; j < jb_cols; ++j)
        d_V_panel[i + j * ldv] = z0;
}

//-----------------------------------------------------------------------------
// Compact WY T-block from S = V^H V and tau.
// T is lower triangular (column j: T(j,j)=tau_j, T(i,j)=-tau_j * (T(i,:)*S(:,j)) for i<j).
// S and T are column-major; S is jb x jb (ld = jb), T is nb x nb (ld = nb).
// Single-thread kernel; nb small so sequential over j is fine.
//-----------------------------------------------------------------------------
__global__ void compute_T_block_s(float* Tb, const float* d_S, const float* d_tau,
                                  int jb, int nb)
{
    extern __shared__ float s_col_f[];
    const int tid = static_cast<int>(threadIdx.x);
    for (int j = 0; j < jb; ++j)
    {
        for (int l = tid; l < j; l += static_cast<int>(blockDim.x))
            s_col_f[l] = d_S[l + j * jb];
        __syncthreads();

        const float tau_j = d_tau[j];
        if (tid == 0)
            Tb[j + j * nb] = tau_j;

        for (int i = tid; i < j; i += static_cast<int>(blockDim.x))
        {
            float sum = 0.f;
            for (int l = 0; l < j; ++l)
                sum += Tb[i + l * nb] * s_col_f[l];
            Tb[i + j * nb] = -tau_j * sum;
        }
        __syncthreads();
    }
}

// DP-accumulation variant: float inputs accumulate in double to reduce
// cancellation during compact WY T-block construction.
__global__ void compute_T_block_s_dpacc(float* Tb, const float* d_S,
                                         const float* d_tau, int jb, int nb)
{
    extern __shared__ float s_col_f_dp[];
    const int tid = static_cast<int>(threadIdx.x);
    for (int j = 0; j < jb; ++j)
    {
        for (int l = tid; l < j; l += static_cast<int>(blockDim.x))
            s_col_f_dp[l] = d_S[l + j * jb];
        __syncthreads();

        const double tau_j = static_cast<double>(d_tau[j]);
        if (tid == 0)
            Tb[j + j * nb] = static_cast<float>(tau_j);

        for (int i = tid; i < j; i += static_cast<int>(blockDim.x))
        {
            double sum = 0.0;
            for (int l = 0; l < j; ++l)
                sum += static_cast<double>(Tb[i + l * nb]) *
                       static_cast<double>(s_col_f_dp[l]);
            Tb[i + j * nb] = static_cast<float>(-tau_j * sum);
        }
        __syncthreads();
    }
}

__global__ void compute_T_block_d(double* Tb, const double* d_S, const double* d_tau,
                                  int jb, int nb)
{
    extern __shared__ double s_col_d[];
    const int tid = static_cast<int>(threadIdx.x);
    for (int j = 0; j < jb; ++j)
    {
        for (int l = tid; l < j; l += static_cast<int>(blockDim.x))
            s_col_d[l] = d_S[l + j * jb];
        __syncthreads();

        const double tau_j = d_tau[j];
        if (tid == 0)
            Tb[j + j * nb] = tau_j;

        for (int i = tid; i < j; i += static_cast<int>(blockDim.x))
        {
            double sum = 0.;
            for (int l = 0; l < j; ++l)
                sum += Tb[i + l * nb] * s_col_d[l];
            Tb[i + j * nb] = -tau_j * sum;
        }
        __syncthreads();
    }
}

// QD-accumulation variant: double inputs accumulate in double-double to
// protect the compact WY T-block coupling from loss of orthogonality.
__global__ void compute_T_block_d_qdacc(double* Tb, const double* d_S,
                                        const double* d_tau, int jb, int nb)
{
    extern __shared__ double s_col_d_qd[];
    const int tid = static_cast<int>(threadIdx.x);
    for (int j = 0; j < jb; ++j)
    {
        for (int l = tid; l < j; l += static_cast<int>(blockDim.x))
            s_col_d_qd[l] = d_S[l + j * jb];
        __syncthreads();

        const double tau_j = d_tau[j];
        if (tid == 0)
            Tb[j + j * nb] = tau_j;

        for (int i = tid; i < j; i += static_cast<int>(blockDim.x))
        {
            QD sum(0.0, 0.0);
            for (int l = 0; l < j; ++l)
            {
                const double a = Tb[i + l * nb];
                const double b = s_col_d_qd[l];
                sum = qd_add_qd(sum, two_prod(a, b));
            }

            // Tb(i,j) = -tau_j * sum
            const QD res = qd_mul_double(sum, -tau_j);
            Tb[i + j * nb] = qd_to_double(res);
        }
        __syncthreads();
    }
}

__global__ void compute_T_block_c(cuComplex* Tb, const cuComplex* d_S, const cuComplex* d_tau,
                                  int jb, int nb)
{
    extern __shared__ cuComplex s_col_c[];
    const int tid = static_cast<int>(threadIdx.x);
    for (int j = 0; j < jb; ++j)
    {
        for (int l = tid; l < j; l += static_cast<int>(blockDim.x))
            s_col_c[l] = d_S[l + j * jb];
        __syncthreads();

        const cuComplex tau_j = d_tau[j];
        if (tid == 0)
            Tb[j + j * nb] = tau_j;

        for (int i = tid; i < j; i += static_cast<int>(blockDim.x))
        {
            cuComplex sum = make_cuComplex(0.f, 0.f);
            for (int l = 0; l < j; ++l)
                sum = cuCaddf(sum, cuCmulf(Tb[i + l * nb], s_col_c[l]));
            Tb[i + j * nb] = cuCmulf(make_cuComplex(-1.f, 0.f), cuCmulf(tau_j, sum));
        }
        __syncthreads();
    }
}

// DP-accumulation variant for complex<float>: accumulate in cuDoubleComplex.
__global__ void compute_T_block_c_dpacc(cuComplex* Tb, const cuComplex* d_S,
                                        const cuComplex* d_tau, int jb, int nb)
{
    extern __shared__ cuComplex s_col_c_dp[];
    const int tid = static_cast<int>(threadIdx.x);
    for (int j = 0; j < jb; ++j)
    {
        for (int l = tid; l < j; l += static_cast<int>(blockDim.x))
            s_col_c_dp[l] = d_S[l + j * jb];
        __syncthreads();

        cuComplex tau_j_f      = d_tau[j];
        if (tid == 0)
            Tb[j + j * nb] = tau_j_f;
        const cuDoubleComplex tau_j =
            make_cuDoubleComplex(static_cast<double>(cuCrealf(tau_j_f)),
                                 static_cast<double>(cuCimagf(tau_j_f)));

        for (int i = tid; i < j; i += static_cast<int>(blockDim.x))
        {
            cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
            for (int l = 0; l < j; ++l)
            {
                cuComplex a_f = Tb[i + l * nb];
                cuComplex b_f = s_col_c_dp[l];
                const cuDoubleComplex a =
                    make_cuDoubleComplex(static_cast<double>(cuCrealf(a_f)),
                                         static_cast<double>(cuCimagf(a_f)));
                const cuDoubleComplex b =
                    make_cuDoubleComplex(static_cast<double>(cuCrealf(b_f)),
                                         static_cast<double>(cuCimagf(b_f)));
                sum = cuCadd(sum, cuCmul(a, b));
            }

            cuDoubleComplex res = cuCmul(tau_j, sum);
            res = make_cuDoubleComplex(-cuCreal(res), -cuCimag(res));
            Tb[i + j * nb] =
                make_cuComplex(static_cast<float>(cuCreal(res)),
                               static_cast<float>(cuCimag(res)));
        }
        __syncthreads();
    }
}

__global__ void compute_T_block_z(cuDoubleComplex* Tb, const cuDoubleComplex* d_S,
                                  const cuDoubleComplex* d_tau, int jb, int nb)
{
    extern __shared__ cuDoubleComplex s_col_z[];
    const int tid = static_cast<int>(threadIdx.x);
    for (int j = 0; j < jb; ++j)
    {
        // Cache S(0:j-1, j) in shared memory once per column j.
        for (int l = tid; l < j; l += static_cast<int>(blockDim.x))
        {
            s_col_z[l] = d_S[l + j * jb];
        }
        __syncthreads();

        const cuDoubleComplex tau_j = d_tau[j];
        if (tid == 0)
            Tb[j + j * nb] = tau_j;

        for (int i = tid; i < j; i += static_cast<int>(blockDim.x))
        {
            cuDoubleComplex sum = make_cuDoubleComplex(0., 0.);
            for (int l = 0; l < j; ++l)
                sum = cuCadd(sum, cuCmul(Tb[i + l * nb], s_col_z[l]));
            Tb[i + j * nb] =
                cuCmul(make_cuDoubleComplex(-1., 0.), cuCmul(tau_j, sum));
        }
        __syncthreads();
    }
}

// QD-accumulation variant for complex<double>: accumulate real/imag in QD.
__global__ void compute_T_block_z_qdacc(cuDoubleComplex* Tb,
                                        const cuDoubleComplex* d_S,
                                        const cuDoubleComplex* d_tau, int jb,
                                        int nb)
{
    extern __shared__ cuDoubleComplex s_col_z_qd[];
    const int tid = static_cast<int>(threadIdx.x);
    for (int j = 0; j < jb; ++j)
    {
        for (int l = tid; l < j; l += static_cast<int>(blockDim.x))
        {
            s_col_z_qd[l] = d_S[l + j * jb];
        }
        __syncthreads();

        const cuDoubleComplex tau_j = d_tau[j];
        if (tid == 0)
            Tb[j + j * nb] = tau_j;

        for (int i = tid; i < j; i += static_cast<int>(blockDim.x))
        {
            ComplexQD sum(QD(0.0, 0.0), QD(0.0, 0.0));
            for (int l = 0; l < j; ++l)
            {
                const cuDoubleComplex a = Tb[i + l * nb];
                const cuDoubleComplex b = s_col_z_qd[l];
                const ComplexQD prod = complex_mul_double_as_qd(a, b);
                sum.re = qd_add_qd(sum.re, prod.re);
                sum.im = qd_add_qd(sum.im, prod.im);
            }

            // Tb(i,j) = -tau_j * sum
            ComplexQD res = complex_mul_doublecomplex_by_qd(tau_j, sum);
            res.re         = qd_neg(res.re);
            res.im         = qd_neg(res.im);

            Tb[i + j * nb] = make_cuDoubleComplex(qd_to_double(res.re),
                                                    qd_to_double(res.im));
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void split_to_hilo_kernel(const T* __restrict__ in_flat,
                                     double* __restrict__ hi,
                                     double* __restrict__ lo,
                                     std::size_t scalar_count)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          static_cast<std::size_t>(threadIdx.x);
    if (i >= scalar_count)
        return;
    const double v = static_cast<double>(in_flat[i]);
    hi[i]          = v;
    lo[i]          = 0.0;
}

__global__ void renorm_hilo_kernel(double* __restrict__ hi,
                                   double* __restrict__ lo,
                                   std::size_t n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          static_cast<std::size_t>(threadIdx.x);
    if (i >= n)
        return;
    QD s = two_sum(hi[i], lo[i]);
    hi[i] = s.hi;
    lo[i] = s.lo;
}

template <typename T>
__global__ void merge_hilo_to_out_kernel(const double* __restrict__ hi,
                                         const double* __restrict__ lo,
                                         T* __restrict__ out_flat,
                                         std::size_t scalar_count)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          static_cast<std::size_t>(threadIdx.x);
    if (i >= scalar_count)
        return;
    out_flat[i] = static_cast<T>(hi[i] + lo[i]);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
