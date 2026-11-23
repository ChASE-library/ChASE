// Ozaki-II: scaling, INT8 packs, INT32 max of C'', max 2^{α'}/2^{β'} (Alg. 2).

#include "utilities.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>

#ifdef CHASE_ENABLE_OZAKI

namespace chase {
namespace linalg {
namespace gemmul8pp {
namespace {

constexpr int kTileDim = 32;

template <typename T>
struct real_type
{
    using type = T;
};
template <>
struct real_type<cuComplex>
{
    using type = float;
};
template <>
struct real_type<cuDoubleComplex>
{
    using type = double;
};
template <typename T>
using real_t = typename real_type<T>::type;

// ---- abs / ilogb (for sft) ----

template <typename T>
__device__ __forceinline__ real_t<T> tabs(T x);

template <>
__device__ __forceinline__ float tabs(float x)
{
    return fabsf(x);
}
template <>
__device__ __forceinline__ double tabs(double x)
{
    return fabs(x);
}
template <>
__device__ __forceinline__ float tabs(cuComplex x)
{
    return fmaxf(fabsf(x.x), fabsf(x.y));
}
template <>
__device__ __forceinline__ double tabs(cuDoubleComplex x)
{
    return fmax(fabs(x.x), fabs(x.y));
}

template <typename R>
__device__ __forceinline__ int tilogb(R x);

template <>
__device__ __forceinline__ int tilogb(float x)
{
    return (x == 0.0f) ? 0 : ilogbf(x);
}
template <>
__device__ __forceinline__ int tilogb(double x)
{
    return (x == 0.0) ? 0 : ilogb(x);
}

template <typename R, int width = 32>
__device__ __forceinline__ R inner_warp_max(R amax)
{
#pragma unroll
    for (int i = width / 2; i > 0; i >>= 1)
        amax = max(amax, __shfl_down_sync(0xffffffffu, amax, i, width));
    return amax;
}

template <typename R>
__device__ __forceinline__ R reduction_max(R max_val, R* smem)
{
    max_val = inner_warp_max(max_val);
    if ((threadIdx.x & 31) == 0)
        smem[threadIdx.x >> 5] = max_val;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        if (threadIdx.x < (blockDim.x >> 5))
            max_val = smem[threadIdx.x];
        max_val = inner_warp_max(max_val);
        if (threadIdx.x == 0)
            smem[0] = max_val;
    }
    __syncthreads();
    return smem[0];
}

template <typename T>
__device__ __forceinline__ real_t<T> find_amax(T const* ptr, unsigned length,
                                               real_t<T>* samax)
{
    using R = real_t<T>;
    R amax  = R(0);
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x)
        amax = max(amax, tabs(ptr[i]));
    return reduction_max(amax, samax);
}

template <typename T>
__device__ __forceinline__ real_t<T> find_amax_tile(unsigned m, unsigned k,
                                                    unsigned row_idx, T const* A,
                                                    std::size_t lda,
                                                    real_t<T> samax[][kTileDim + 1])
{
    using R = real_t<T>;
    R amax  = R(0);
    if (row_idx < m)
    {
        T const* row = A + row_idx;
        for (unsigned col = threadIdx.y; col < k; col += blockDim.y)
            amax = max(amax, tabs(row[col * lda]));
    }
    samax[threadIdx.y][threadIdx.x] = amax;
    __syncthreads();
    amax = samax[threadIdx.x][threadIdx.y];
    return inner_warp_max<R, kTileDim>(amax);
}

// ---- ceil(|x| * 2^sft) → int8 (GEMMul8 store.hpp style) ----

__device__ __forceinline__ std::int8_t ceil_scalbn_8i(float in, int sft)
{
    return static_cast<std::int8_t>(
        __float2int_ru(scalbnf(fabsf(in), sft)));
}
__device__ __forceinline__ std::int8_t ceil_scalbn_8i(double in, int sft)
{
    return static_cast<std::int8_t>(
        __double2int_ru(scalbn(fabs(in), sft)));
}

__device__ __forceinline__ std::int8_t upperbound_lo(float in, int sft)
{
    return ceil_scalbn_8i(in, sft);
}
__device__ __forceinline__ std::int8_t upperbound_lo(double in, int sft)
{
    return ceil_scalbn_8i(in, sft);
}
__device__ __forceinline__ char2 upperbound_lo(cuComplex in, int sft)
{
    return make_char2(ceil_scalbn_8i(in.x, sft), ceil_scalbn_8i(in.y, sft));
}
__device__ __forceinline__ char2 upperbound_lo(cuDoubleComplex in, int sft)
{
    return make_char2(ceil_scalbn_8i(in.x, sft), ceil_scalbn_8i(in.y, sft));
}

// Scale entry by inv_e (= 1/|e|) without rewriting the matrix buffer.
__device__ __forceinline__ float scale_inv_e(float x, float inv_e)
{
    return x * inv_e;
}
__device__ __forceinline__ double scale_inv_e(double x, double inv_e)
{
    return x * inv_e;
}
__device__ __forceinline__ cuComplex scale_inv_e(cuComplex x, float inv_e)
{
    return make_cuComplex(x.x * inv_e, x.y * inv_e);
}
__device__ __forceinline__ cuDoubleComplex scale_inv_e(cuDoubleComplex x,
                                                       double inv_e)
{
    return make_cuDoubleComplex(x.x * inv_e, x.y * inv_e);
}

// ---- sft kernels ----

template <typename T>
__global__ void row_sft_kernel(unsigned m, unsigned k, T const* A,
                               std::size_t lda, std::int16_t* sft,
                               real_t<T> inv_e)
{
    using R = real_t<T>;
    __shared__ R samax[kTileDim][kTileDim + 1];
    unsigned row = blockIdx.x * kTileDim + threadIdx.x;
    R const amax = find_amax_tile(m, k, row, A, lda, samax) * inv_e;
    row          = blockIdx.x * kTileDim + threadIdx.y;
    if (row < m && threadIdx.x == 0)
        sft[row] = static_cast<std::int16_t>(kMaxUFP - tilogb(amax));
}

template <typename T>
__global__ void col_sft_kernel(unsigned k, T const* V, std::size_t ldv,
                               std::int16_t* sft)
{
    using R = real_t<T>;
    __shared__ R samax[32];
    unsigned const j = blockIdx.x;
    R const amax     = find_amax(V + j * ldv, k, samax);
    if (threadIdx.x == 0)
        sft[j] = static_cast<std::int16_t>(kMaxUFP - tilogb(amax));
}

// ---- extract_A_lo (row-scaled pack) ----

template <typename T>
__global__ void extract_A_lo_real_kernel(unsigned m, unsigned k, T const* A,
                                         std::size_t lda, std::int16_t const* sftA,
                                         std::int8_t* A_lo, std::size_t lda_lo,
                                         real_t<T> inv_e)
{
    __shared__ std::int8_t tile[kTileDim][kTileDim + 1];

    unsigned const rowBase = blockIdx.x * kTileDim;
    unsigned const colBase = blockIdx.y * kTileDim;
    unsigned const in_row  = rowBase + threadIdx.x;
    unsigned const in_col  = colBase + threadIdx.y;

    int const sft = (in_row < m) ? sftA[in_row] : 0;
    T const Atmp =
        (in_row < m && in_col < k) ? A[in_col * lda + in_row] : T(0);
    tile[threadIdx.y][threadIdx.x] =
        upperbound_lo(scale_inv_e(Atmp, inv_e), sft);
    __syncthreads();

    unsigned const out_row = colBase + threadIdx.x;
    unsigned const out_col = rowBase + threadIdx.y;
    if (out_col >= m || out_row >= lda_lo)
        return;
    A_lo[out_col * lda_lo + out_row] = tile[threadIdx.x][threadIdx.y];
}

template <typename T>
__global__ void extract_A_lo_cplx_kernel(unsigned m, unsigned k, T const* A,
                                         std::size_t lda, std::int16_t const* sftA,
                                         std::int8_t* A_lo_re, std::int8_t* A_lo_im,
                                         std::size_t lda_lo, real_t<T> inv_e)
{
    __shared__ char2 tile[kTileDim][kTileDim + 1];

    unsigned const rowBase = blockIdx.x * kTileDim;
    unsigned const colBase = blockIdx.y * kTileDim;
    unsigned const in_row  = rowBase + threadIdx.x;
    unsigned const in_col  = colBase + threadIdx.y;

    int const sft = (in_row < m) ? sftA[in_row] : 0;
    T const zero{};
    T const Atmp =
        (in_row < m && in_col < k) ? A[in_col * lda + in_row] : zero;
    tile[threadIdx.y][threadIdx.x] =
        upperbound_lo(scale_inv_e(Atmp, inv_e), sft);
    __syncthreads();

    unsigned const out_row = colBase + threadIdx.x;
    unsigned const out_col = rowBase + threadIdx.y;
    if (out_col >= m || out_row >= lda_lo)
        return;
    char2 const v                    = tile[threadIdx.x][threadIdx.y];
    A_lo_re[out_col * lda_lo + out_row] = v.x;
    A_lo_im[out_col * lda_lo + out_row] = v.y;
}

// ---- extract_B_lo given sft (col-scaled pack) ----

template <typename T>
__global__ void extract_B_lo_real_kernel(unsigned k, T const* B, std::size_t ldb,
                                         std::int16_t const* sftB, std::int8_t* B_lo,
                                         std::size_t ldb_lo)
{
    unsigned const j   = blockIdx.x;
    int const sft      = sftB[j];
    T const* in        = B + j * ldb;
    std::int8_t* out   = B_lo + j * ldb_lo;
    for (unsigned i = threadIdx.x; i < ldb_lo; i += blockDim.x)
    {
        T const v = (i < k) ? in[i] : T(0);
        out[i]    = (i < k) ? upperbound_lo(v, sft) : std::int8_t(0);
    }
}

template <typename T>
__global__ void extract_B_lo_cplx_kernel(unsigned k, T const* B, std::size_t ldb,
                                         std::int16_t const* sftB,
                                         std::int8_t* B_lo_re, std::int8_t* B_lo_im,
                                         std::size_t ldb_lo)
{
    unsigned const j = blockIdx.x;
    int const sft    = sftB[j];
    T const* in      = B + j * ldb;
    std::int8_t* re  = B_lo_re + j * ldb_lo;
    std::int8_t* im  = B_lo_im + j * ldb_lo;
    for (unsigned i = threadIdx.x; i < ldb_lo; i += blockDim.x)
    {
        if (i < k)
        {
            char2 const v = upperbound_lo(in[i], sft);
            re[i]         = v.x;
            im[i]         = v.y;
        }
        else
        {
            re[i] = 0;
            im[i] = 0;
        }
    }
}

// ---- host launchers ----

template <typename T>
void compute_row_sft_impl(T const* A, std::size_t lda, std::size_t m,
                          std::size_t k, std::int16_t* sft, double inv_e,
                          cudaStream_t stream)
{
    using R                = real_t<T>;
    dim3 const threads(kTileDim, kTileDim);
    unsigned const blocks = static_cast<unsigned>((m + kTileDim - 1) / kTileDim);
    row_sft_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<unsigned>(m), static_cast<unsigned>(k), A, lda, sft,
        static_cast<R>(inv_e));
}

template <typename T>
void compute_col_sft_impl(T const* V, std::size_t ldv, std::size_t k,
                          std::size_t n, std::int16_t* sft, cudaStream_t stream)
{
    col_sft_kernel<T><<<static_cast<unsigned>(n), 256, 0, stream>>>(
        static_cast<unsigned>(k), V, ldv, sft);
}

template <typename T>
void pack_A_lo_real_impl(T const* A, std::size_t lda, std::size_t m,
                         std::size_t k, std::int16_t const* sftA,
                         std::int8_t* A_lo, std::size_t lda_lo, double inv_e,
                         cudaStream_t stream)
{
    using R = real_t<T>;
    dim3 const threads(kTileDim, kTileDim);
    dim3 const grid(static_cast<unsigned>((m + kTileDim - 1) / kTileDim),
                    static_cast<unsigned>((lda_lo + kTileDim - 1) / kTileDim));
    extract_A_lo_real_kernel<T><<<grid, threads, 0, stream>>>(
        static_cast<unsigned>(m), static_cast<unsigned>(k), A, lda, sftA, A_lo,
        lda_lo, static_cast<R>(inv_e));
}

template <typename T>
void pack_A_lo_cplx_impl(T const* A, std::size_t lda, std::size_t m,
                         std::size_t k, std::int16_t const* sftA,
                         std::int8_t* A_lo_re, std::int8_t* A_lo_im,
                         std::size_t lda_lo, double inv_e, cudaStream_t stream)
{
    using R = real_t<T>;
    dim3 const threads(kTileDim, kTileDim);
    dim3 const grid(static_cast<unsigned>((m + kTileDim - 1) / kTileDim),
                    static_cast<unsigned>((lda_lo + kTileDim - 1) / kTileDim));
    extract_A_lo_cplx_kernel<T><<<grid, threads, 0, stream>>>(
        static_cast<unsigned>(m), static_cast<unsigned>(k), A, lda, sftA,
        A_lo_re, A_lo_im, lda_lo, static_cast<R>(inv_e));
}

template <typename T>
void pack_B_lo_real_impl(T const* B, std::size_t ldb, std::size_t k,
                         std::size_t n, std::int16_t const* sftB,
                         std::int8_t* B_lo, std::size_t ldb_lo, cudaStream_t stream)
{
    extract_B_lo_real_kernel<T><<<static_cast<unsigned>(n), 256, 0, stream>>>(
        static_cast<unsigned>(k), B, ldb, sftB, B_lo, ldb_lo);
}

template <typename T>
void pack_B_lo_cplx_impl(T const* B, std::size_t ldb, std::size_t k,
                         std::size_t n, std::int16_t const* sftB,
                         std::int8_t* B_lo_re, std::int8_t* B_lo_im,
                         std::size_t ldb_lo, cudaStream_t stream)
{
    extract_B_lo_cplx_kernel<T>
        <<<static_cast<unsigned>(n), 256, 0, stream>>>(
            static_cast<unsigned>(k), B, ldb, sftB, B_lo_re, B_lo_im, ldb_lo);
}

// ---- INT32 max of C'' (PAR find_max.hpp style) ----

__device__ __forceinline__ std::int32_t find_max(std::int32_t const* C,
                                                  unsigned length,
                                                  std::int32_t* samax)
{
    std::int32_t amax = 0;
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x)
        amax = max(amax, C[i]);
    return reduction_max(amax, samax);
}

__device__ __forceinline__ std::int32_t find_max_tile(
    unsigned m, unsigned n, unsigned row_idx, std::int32_t const* C,
    std::size_t ldc, std::int32_t samax[][kTileDim + 1])
{
    std::int32_t amax = 0;
    if (row_idx < m)
    {
        for (unsigned col = threadIdx.y; col < n; col += blockDim.y)
            amax = max(amax, C[col * ldc + row_idx]);
    }
    samax[threadIdx.y][threadIdx.x] = amax;
    __syncthreads();
    amax = samax[threadIdx.x][threadIdx.y];
    return inner_warp_max<std::int32_t, kTileDim>(amax);
}

__device__ __forceinline__ std::int32_t find_max_complex(
    std::int32_t const* C1, std::int32_t const* C2, unsigned length,
    std::int32_t* samax)
{
    std::int32_t amax = 0;
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x)
    {
        std::int32_t const t1 = C1[i];
        std::int32_t const t2 = C2[i];
        std::int32_t const t3 = t1 + t2;
        amax                  = max(amax, (t3 > t2) ? t3 : t2);
    }
    return reduction_max(amax, samax);
}

__device__ __forceinline__ std::int32_t find_max_tile_complex(
    unsigned m, unsigned n, unsigned row_idx, std::int32_t const* C1,
    std::int32_t const* C2, std::size_t ldc,
    std::int32_t samax[][kTileDim + 1])
{
    std::int32_t amax = 0;
    if (row_idx < m)
    {
        for (unsigned col = threadIdx.y; col < n; col += blockDim.y)
        {
            std::int32_t const t1 = C1[col * ldc + row_idx];
            std::int32_t const t2 = C2[col * ldc + row_idx];
            std::int32_t const t3 = t1 + t2;
            amax                  = max(amax, (t3 > t2) ? t3 : t2);
        }
    }
    samax[threadIdx.y][threadIdx.x] = amax;
    __syncthreads();
    amax = samax[threadIdx.x][threadIdx.y];
    return inner_warp_max<std::int32_t, kTileDim>(amax);
}

__global__ void row_max_C_kernel(unsigned m, unsigned n, std::int32_t const* C,
                                 std::size_t ldc, std::int32_t* amax_row)
{
    __shared__ std::int32_t samax[kTileDim][kTileDim + 1];
    unsigned row              = blockIdx.x * kTileDim + threadIdx.x;
    std::int32_t const amax   = find_max_tile(m, n, row, C, ldc, samax);
    row                       = blockIdx.x * kTileDim + threadIdx.y;
    if (row < m && threadIdx.x == 0)
        amax_row[row] = amax;
}

__global__ void col_max_C_kernel(unsigned m, std::int32_t const* C,
                                 std::size_t ldc, std::int32_t* amax_col)
{
    __shared__ std::int32_t shm[32];
    unsigned const j            = blockIdx.x;
    std::int32_t const amax     = find_max(C + j * ldc, m, shm);
    if (threadIdx.x == 0)
        amax_col[j] = amax;
}

__global__ void row_max_C_cplx_kernel(unsigned m, unsigned n,
                                      std::int32_t const* C1,
                                      std::int32_t const* C2, std::size_t ldc,
                                      std::int32_t* amax_row)
{
    __shared__ std::int32_t samax[kTileDim][kTileDim + 1];
    unsigned row            = blockIdx.x * kTileDim + threadIdx.x;
    std::int32_t const amax = find_max_tile_complex(m, n, row, C1, C2, ldc, samax);
    row                     = blockIdx.x * kTileDim + threadIdx.y;
    if (row < m && threadIdx.x == 0)
        amax_row[row] = amax;
}

__global__ void col_max_C_cplx_kernel(unsigned m, std::int32_t const* C1,
                                      std::int32_t const* C2, std::size_t ldc,
                                      std::int32_t* amax_col)
{
    __shared__ std::int32_t shm[32];
    unsigned const j        = blockIdx.x;
    std::int32_t const amax = find_max_complex(C1 + j * ldc, C2 + j * ldc, m, shm);
    if (threadIdx.x == 0)
        amax_col[j] = amax;
}

// α' = (kMaxUFP - sft) + ½ log2(amax_C); write 2^{α'} (positive).
__global__ void ozaki_two_pow_prime_kernel(std::int16_t const* sft,
                                           std::int32_t const* amax_C,
                                           unsigned n, double* two_pow)
{
    unsigned const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    double const alpha = static_cast<double>(kMaxUFP - static_cast<int>(sft[i]));
    double const e     = static_cast<double>(amax_C[i]);
    double const ap =
        (e > 0.0) ? (alpha + 0.5 * log2(e)) : alpha;
    two_pow[i] = exp2(ap);
}

} // namespace

// ---- sft API ----

template <>
void compute_row_sft<float>(float const* A, std::size_t lda, std::size_t m,
                            std::size_t k, std::int16_t* sft, double inv_e,
                            cudaStream_t stream)
{
    compute_row_sft_impl(A, lda, m, k, sft, inv_e, stream);
}
template <>
void compute_row_sft<double>(double const* A, std::size_t lda, std::size_t m,
                             std::size_t k, std::int16_t* sft, double inv_e,
                             cudaStream_t stream)
{
    compute_row_sft_impl(A, lda, m, k, sft, inv_e, stream);
}
template <>
void compute_row_sft<std::complex<float>>(std::complex<float> const* A,
                                          std::size_t lda, std::size_t m,
                                          std::size_t k, std::int16_t* sft,
                                          double inv_e, cudaStream_t stream)
{
    compute_row_sft_impl(reinterpret_cast<cuComplex const*>(A), lda, m, k, sft,
                         inv_e, stream);
}
template <>
void compute_row_sft<std::complex<double>>(std::complex<double> const* A,
                                           std::size_t lda, std::size_t m,
                                           std::size_t k, std::int16_t* sft,
                                           double inv_e, cudaStream_t stream)
{
    compute_row_sft_impl(reinterpret_cast<cuDoubleComplex const*>(A), lda, m, k,
                         sft, inv_e, stream);
}

template <>
void compute_col_sft<float>(float const* V, std::size_t ldv, std::size_t k,
                            std::size_t n, std::int16_t* sft,
                            cudaStream_t stream)
{
    compute_col_sft_impl(V, ldv, k, n, sft, stream);
}
template <>
void compute_col_sft<double>(double const* V, std::size_t ldv, std::size_t k,
                             std::size_t n, std::int16_t* sft,
                             cudaStream_t stream)
{
    compute_col_sft_impl(V, ldv, k, n, sft, stream);
}
template <>
void compute_col_sft<std::complex<float>>(std::complex<float> const* V,
                                          std::size_t ldv, std::size_t k,
                                          std::size_t n, std::int16_t* sft,
                                          cudaStream_t stream)
{
    compute_col_sft_impl(reinterpret_cast<cuComplex const*>(V), ldv, k, n, sft,
                         stream);
}
template <>
void compute_col_sft<std::complex<double>>(std::complex<double> const* V,
                                           std::size_t ldv, std::size_t k,
                                           std::size_t n, std::int16_t* sft,
                                           cudaStream_t stream)
{
    compute_col_sft_impl(reinterpret_cast<cuDoubleComplex const*>(V), ldv, k, n,
                         sft, stream);
}

// ---- pack API (real: one plane) ----

template <>
void pack_A_lo<float>(float const* A, std::size_t lda, std::size_t m,
                      std::size_t k, std::int16_t const* sftA, std::int8_t* A_lo,
                      std::size_t lda_lo, double inv_e, cudaStream_t stream)
{
    pack_A_lo_real_impl(A, lda, m, k, sftA, A_lo, lda_lo, inv_e, stream);
}
template <>
void pack_A_lo<double>(double const* A, std::size_t lda, std::size_t m,
                       std::size_t k, std::int16_t const* sftA, std::int8_t* A_lo,
                       std::size_t lda_lo, double inv_e, cudaStream_t stream)
{
    pack_A_lo_real_impl(A, lda, m, k, sftA, A_lo, lda_lo, inv_e, stream);
}

template <>
void pack_B_lo<float>(float const* B, std::size_t ldb, std::size_t k,
                      std::size_t n, std::int16_t const* sftB, std::int8_t* B_lo,
                      std::size_t ldb_lo, cudaStream_t stream)
{
    pack_B_lo_real_impl(B, ldb, k, n, sftB, B_lo, ldb_lo, stream);
}
template <>
void pack_B_lo<double>(double const* B, std::size_t ldb, std::size_t k,
                       std::size_t n, std::int16_t const* sftB, std::int8_t* B_lo,
                       std::size_t ldb_lo, cudaStream_t stream)
{
    pack_B_lo_real_impl(B, ldb, k, n, sftB, B_lo, ldb_lo, stream);
}

// ---- pack API (complex: Re/Im planes) ----

template <>
void pack_A_lo<std::complex<float>>(std::complex<float> const* A, std::size_t lda,
                                    std::size_t m, std::size_t k,
                                    std::int16_t const* sftA, std::int8_t* A_lo_re,
                                    std::int8_t* A_lo_im, std::size_t lda_lo,
                                    double inv_e, cudaStream_t stream)
{
    pack_A_lo_cplx_impl(reinterpret_cast<cuComplex const*>(A), lda, m, k, sftA,
                        A_lo_re, A_lo_im, lda_lo, inv_e, stream);
}
template <>
void pack_A_lo<std::complex<double>>(std::complex<double> const* A,
                                     std::size_t lda, std::size_t m, std::size_t k,
                                     std::int16_t const* sftA, std::int8_t* A_lo_re,
                                     std::int8_t* A_lo_im, std::size_t lda_lo,
                                     double inv_e, cudaStream_t stream)
{
    pack_A_lo_cplx_impl(reinterpret_cast<cuDoubleComplex const*>(A), lda, m, k,
                        sftA, A_lo_re, A_lo_im, lda_lo, inv_e, stream);
}

template <>
void pack_B_lo<std::complex<float>>(std::complex<float> const* B, std::size_t ldb,
                                    std::size_t k, std::size_t n,
                                    std::int16_t const* sftB, std::int8_t* B_lo_re,
                                    std::int8_t* B_lo_im, std::size_t ldb_lo,
                                    cudaStream_t stream)
{
    pack_B_lo_cplx_impl(reinterpret_cast<cuComplex const*>(B), ldb, k, n, sftB,
                        B_lo_re, B_lo_im, ldb_lo, stream);
}
template <>
void pack_B_lo<std::complex<double>>(std::complex<double> const* B,
                                     std::size_t ldb, std::size_t k, std::size_t n,
                                     std::int16_t const* sftB, std::int8_t* B_lo_re,
                                     std::int8_t* B_lo_im, std::size_t ldb_lo,
                                     cudaStream_t stream)
{
    pack_B_lo_cplx_impl(reinterpret_cast<cuDoubleComplex const*>(B), ldb, k, n,
                        sftB, B_lo_re, B_lo_im, ldb_lo, stream);
}

// ---- INT32 max of C'' API ----

void compute_row_max_C(std::int32_t const* C, std::size_t ldc, std::size_t m,
                       std::size_t n, std::int32_t* amax_row,
                       cudaStream_t stream)
{
    dim3 const threads(kTileDim, kTileDim);
    unsigned const grid =
        static_cast<unsigned>((m + kTileDim - 1) / kTileDim);
    row_max_C_kernel<<<grid, threads, 0, stream>>>(static_cast<unsigned>(m),
                                                   static_cast<unsigned>(n), C,
                                                   ldc, amax_row);
}

void compute_col_max_C(std::int32_t const* C, std::size_t ldc, std::size_t m,
                       std::size_t n, std::int32_t* amax_col,
                       cudaStream_t stream)
{
    col_max_C_kernel<<<static_cast<unsigned>(n), 256, 0, stream>>>(
        static_cast<unsigned>(m), C, ldc, amax_col);
}

void compute_row_max_C(std::int32_t const* C1, std::int32_t const* C2,
                       std::size_t ldc, std::size_t m, std::size_t n,
                       std::int32_t* amax_row, cudaStream_t stream)
{
    dim3 const threads(kTileDim, kTileDim);
    unsigned const grid =
        static_cast<unsigned>((m + kTileDim - 1) / kTileDim);
    row_max_C_cplx_kernel<<<grid, threads, 0, stream>>>(
        static_cast<unsigned>(m), static_cast<unsigned>(n), C1, C2, ldc,
        amax_row);
}

void compute_col_max_C(std::int32_t const* C1, std::int32_t const* C2,
                       std::size_t ldc, std::size_t m, std::size_t n,
                       std::int32_t* amax_col, cudaStream_t stream)
{
    col_max_C_cplx_kernel<<<static_cast<unsigned>(n), 256, 0, stream>>>(
        static_cast<unsigned>(m), C1, C2, ldc, amax_col);
}

// ---- max 2^{α'} / max 2^{β'} API ----

void compute_ozaki_two_pow_primes(std::int16_t const* sft,
                                  std::int32_t const* amax_C, std::size_t n,
                                  double* two_pow, cudaStream_t stream)
{
    unsigned const block = 256;
    unsigned const grid  = static_cast<unsigned>((n + block - 1) / block);
    ozaki_two_pow_prime_kernel<<<grid, block, 0, stream>>>(
        sft, amax_C, static_cast<unsigned>(n), two_pow);
}

double max_two_pow_ozaki_prime(cublasHandle_t handle, std::int16_t const* sft,
                               std::int32_t const* amax_C, std::size_t n,
                               double* d_two_pow)
{
    if (n == 0)
        return 0.0;

    cudaStream_t stream = nullptr;
    cublasGetStream(handle, &stream);
    compute_ozaki_two_pow_primes(sft, amax_C, n, d_two_pow, stream);

    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    int idx1 = 0;
    cublasStatus_t const iamax_status =
        cublasIdamax(handle, static_cast<int>(n), d_two_pow, 1, &idx1);
    assert(iamax_status == CUBLAS_STATUS_SUCCESS);
    (void)iamax_status;
    if (idx1 < 1)
        return 0.0;

    double h_max = 0.0;
    cudaMemcpyAsync(&h_max, d_two_pow + (static_cast<std::size_t>(idx1) - 1),
                    sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h_max;
}

double nrm2_two_pow_ozaki_prime(cublasHandle_t handle, std::int16_t const* sft,
                                std::int32_t const* amax_C, std::size_t n,
                                double* d_two_pow)
{
    if (n == 0)
        return 0.0;

    cudaStream_t stream = nullptr;
    cublasGetStream(handle, &stream);
    compute_ozaki_two_pow_primes(sft, amax_C, n, d_two_pow, stream);

    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    double h_nrm2 = 0.0;
    cublasStatus_t const nrm2_status =
        cublasDnrm2(handle, static_cast<int>(n), d_two_pow, 1, &h_nrm2);
    assert(nrm2_status == CUBLAS_STATUS_SUCCESS);
    (void)nrm2_status;
    return h_nrm2;
}

namespace {

// cuBLAS INT8 GEMM requires m, n, k and the leading dimensions to be
// multiples of 4, and the matrix pointers to be 16-byte aligned.
constexpr std::size_t kInt8Align = 16;

std::size_t align_up(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

std::size_t pad_to_multiple_of_4(std::size_t n)
{
    return (n + 3) & ~std::size_t{3};
}

struct Int8ProbeLayout
{
    std::size_t m_pad;
    std::size_t n_pad;
    std::size_t k_pad;
    std::size_t sft_a;
    std::size_t sft_b;
    std::size_t a_lo;
    std::size_t a_lo_bytes;
    std::size_t b_lo;
    std::size_t b_lo_bytes;
    std::size_t c;
    std::size_t c_bytes;
    std::size_t amax_row;
    std::size_t amax_col;
    std::size_t two_pow;
    std::size_t bytes;
};

Int8ProbeLayout int8_probe_layout(std::size_t m, std::size_t n, std::size_t k,
                                  bool is_complex)
{
    Int8ProbeLayout layout{};
    layout.m_pad = pad_to_multiple_of_4(m);
    layout.n_pad = pad_to_multiple_of_4(n);
    layout.k_pad = pad_to_multiple_of_4(k);

    std::size_t const planes = is_complex ? 2 : 1;
    std::size_t off          = 0;

    layout.sft_a = off;
    off = align_up(off + m * sizeof(std::int16_t), kInt8Align);

    layout.sft_b = off;
    off = align_up(off + n * sizeof(std::int16_t), kInt8Align);

    layout.a_lo       = off;
    layout.a_lo_bytes = layout.m_pad * layout.k_pad;
    off = align_up(off + planes * layout.a_lo_bytes, kInt8Align);

    layout.b_lo       = off;
    layout.b_lo_bytes = layout.k_pad * layout.n_pad;
    off = align_up(off + planes * layout.b_lo_bytes, kInt8Align);

    layout.c       = off;
    layout.c_bytes = layout.m_pad * layout.n_pad * sizeof(std::int32_t);
    off = align_up(off + planes * layout.c_bytes, kInt8Align);

    layout.amax_row = off;
    off = align_up(off + m * sizeof(std::int32_t), kInt8Align);

    layout.amax_col = off;
    off = align_up(off + n * sizeof(std::int32_t), kInt8Align);

    layout.two_pow = off;
    std::size_t const tmax = m > n ? m : n;
    off += tmax * sizeof(double);

    layout.bytes = off;
    return layout;
}

void zero_device(void* ptr, std::size_t bytes, cudaStream_t stream)
{
    if (bytes == 0)
    {
        return;
    }
    cudaError_t const memset_status = cudaMemsetAsync(ptr, 0, bytes, stream);
    assert(memset_status == cudaSuccess);
    (void)memset_status;
}

std::size_t probe_work_bytes(std::size_t m, std::size_t n, std::size_t k,
                             bool is_complex)
{
    return int8_probe_layout(m, n, k, is_complex).bytes;
}

void gemm_i8_tn(cublasHandle_t handle, std::size_t m, std::size_t n,
                std::size_t k, std::int8_t const* A_lo, std::int8_t const* B_lo,
                std::int32_t* C)
{
    std::int32_t const one  = 1;
    std::int32_t const zero = 0;
    cublasPointerMode_t prev = CUBLAS_POINTER_MODE_DEVICE;
    cublasGetPointerMode(handle, &prev);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasStatus_t const gemm_status = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(m),
        static_cast<int>(n), static_cast<int>(k), &one, A_lo, CUDA_R_8I,
        static_cast<int>(k), B_lo, CUDA_R_8I, static_cast<int>(k), &zero, C,
        CUDA_R_32I, static_cast<int>(m), CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT);
    assert(gemm_status == CUBLAS_STATUS_SUCCESS);
    (void)gemm_status;
    cublasSetPointerMode(handle, prev);
}

template <typename T>
void ozaki_compute_scaling_factors_error_bound_real(
    cublasHandle_t handle, T const* A, std::size_t lda, std::size_t m,
    std::size_t k, T const* B, std::size_t ldb, std::size_t n, void* work,
    std::size_t work_bytes, double* h_nrm2_a, double* h_max2_b, double inv_e)
{
    Int8ProbeLayout const layout = int8_probe_layout(m, n, k, false);
    assert(layout.bytes <= work_bytes);

    char* const base = static_cast<char*>(work);
    auto* sftA       = reinterpret_cast<std::int16_t*>(base + layout.sft_a);
    auto* sftB       = reinterpret_cast<std::int16_t*>(base + layout.sft_b);
    auto* A_lo       = reinterpret_cast<std::int8_t*>(base + layout.a_lo);
    auto* B_lo       = reinterpret_cast<std::int8_t*>(base + layout.b_lo);
    auto* C          = reinterpret_cast<std::int32_t*>(base + layout.c);
    auto* amax_row   = reinterpret_cast<std::int32_t*>(base + layout.amax_row);
    auto* amax_col   = reinterpret_cast<std::int32_t*>(base + layout.amax_col);
    auto* d_two_pow  = reinterpret_cast<double*>(base + layout.two_pow);

    cudaStream_t stream = nullptr;
    cublasGetStream(handle, &stream);

    compute_row_sft(A, lda, m, k, sftA, inv_e, stream);
    compute_col_sft(B, ldb, k, n, sftB, stream);
    zero_device(A_lo, layout.a_lo_bytes, stream);
    zero_device(B_lo, layout.b_lo_bytes, stream);
    pack_A_lo(A, lda, m, k, sftA, A_lo, layout.k_pad, inv_e, stream);
    pack_B_lo(B, ldb, k, n, sftB, B_lo, layout.k_pad, stream);

    gemm_i8_tn(handle, layout.m_pad, layout.n_pad, layout.k_pad, A_lo, B_lo, C);

    compute_row_max_C(C, layout.m_pad, m, n, amax_row, stream);
    compute_col_max_C(C, layout.m_pad, m, n, amax_col, stream);

    *h_nrm2_a = nrm2_two_pow_ozaki_prime(handle, sftA, amax_row, m, d_two_pow);
    *h_max2_b = max_two_pow_ozaki_prime(handle, sftB, amax_col, n, d_two_pow);
}

template <typename T>
void ozaki_compute_scaling_factors_error_bound_cplx(
    cublasHandle_t handle, T const* A, std::size_t lda, std::size_t m,
    std::size_t k, T const* B, std::size_t ldb, std::size_t n, void* work,
    std::size_t work_bytes, double* h_nrm2_a, double* h_max2_b, double inv_e)
{
    Int8ProbeLayout const layout = int8_probe_layout(m, n, k, true);
    assert(layout.bytes <= work_bytes);

    char* const base = static_cast<char*>(work);
    auto* sftA       = reinterpret_cast<std::int16_t*>(base + layout.sft_a);
    auto* sftB       = reinterpret_cast<std::int16_t*>(base + layout.sft_b);
    auto* A_lo_re    = reinterpret_cast<std::int8_t*>(base + layout.a_lo);
    auto* A_lo_im    = A_lo_re + layout.a_lo_bytes;
    auto* B_lo_re    = reinterpret_cast<std::int8_t*>(base + layout.b_lo);
    auto* B_lo_im    = B_lo_re + layout.b_lo_bytes;
    auto* C1         = reinterpret_cast<std::int32_t*>(base + layout.c);
    auto* C2         = C1 + (layout.c_bytes / sizeof(std::int32_t));
    auto* amax_row   = reinterpret_cast<std::int32_t*>(base + layout.amax_row);
    auto* amax_col   = reinterpret_cast<std::int32_t*>(base + layout.amax_col);
    auto* d_two_pow  = reinterpret_cast<double*>(base + layout.two_pow);

    cudaStream_t stream = nullptr;
    cublasGetStream(handle, &stream);

    compute_row_sft(A, lda, m, k, sftA, inv_e, stream);
    compute_col_sft(B, ldb, k, n, sftB, stream);
    zero_device(A_lo_re, 2 * layout.a_lo_bytes, stream);
    zero_device(B_lo_re, 2 * layout.b_lo_bytes, stream);
    pack_A_lo(A, lda, m, k, sftA, A_lo_re, A_lo_im, layout.k_pad, inv_e, stream);
    pack_B_lo(B, ldb, k, n, sftB, B_lo_re, B_lo_im, layout.k_pad, stream);

    gemm_i8_tn(handle, layout.m_pad, layout.n_pad, layout.k_pad, A_lo_re, B_lo_re, C1);
    gemm_i8_tn(handle, layout.m_pad, layout.n_pad, layout.k_pad, A_lo_im, B_lo_im, C2);

    compute_row_max_C(C1, C2, layout.m_pad, m, n, amax_row, stream);
    compute_col_max_C(C1, C2, layout.m_pad, m, n, amax_col, stream);

    *h_nrm2_a = nrm2_two_pow_ozaki_prime(handle, sftA, amax_row, m, d_two_pow);
    *h_max2_b = max_two_pow_ozaki_prime(handle, sftB, amax_col, n, d_two_pow);
}

} // namespace

template <>
std::size_t ozaki_scaling_factors_error_bound_work_bytes<float>(std::size_t m,
                                                                std::size_t n,
                                                                std::size_t k)
{
    return probe_work_bytes(m, n, k, false);
}
template <>
std::size_t ozaki_scaling_factors_error_bound_work_bytes<double>(std::size_t m,
                                                                 std::size_t n,
                                                                 std::size_t k)
{
    return probe_work_bytes(m, n, k, false);
}
template <>
std::size_t ozaki_scaling_factors_error_bound_work_bytes<std::complex<float>>(
    std::size_t m, std::size_t n, std::size_t k)
{
    return probe_work_bytes(m, n, k, true);
}
template <>
std::size_t ozaki_scaling_factors_error_bound_work_bytes<std::complex<double>>(
    std::size_t m, std::size_t n, std::size_t k)
{
    return probe_work_bytes(m, n, k, true);
}

template <>
void ozaki_compute_scaling_factors_error_bound<float>(
    cublasHandle_t handle, float const* A, std::size_t lda, std::size_t m,
    std::size_t k, float const* B, std::size_t ldb, std::size_t n, void* work,
    std::size_t work_bytes, double* h_nrm2_a, double* h_max2_b, double inv_e)
{
    ozaki_compute_scaling_factors_error_bound_real(handle, A, lda, m, k, B, ldb,
                                                   n, work, work_bytes,
                                                   h_nrm2_a, h_max2_b, inv_e);
}

template <>
void ozaki_compute_scaling_factors_error_bound<double>(
    cublasHandle_t handle, double const* A, std::size_t lda, std::size_t m,
    std::size_t k, double const* B, std::size_t ldb, std::size_t n, void* work,
    std::size_t work_bytes, double* h_nrm2_a, double* h_max2_b, double inv_e)
{
    ozaki_compute_scaling_factors_error_bound_real(handle, A, lda, m, k, B, ldb,
                                                   n, work, work_bytes,
                                                   h_nrm2_a, h_max2_b, inv_e);
}

template <>
void ozaki_compute_scaling_factors_error_bound<std::complex<float>>(
    cublasHandle_t handle, std::complex<float> const* A, std::size_t lda,
    std::size_t m, std::size_t k, std::complex<float> const* B, std::size_t ldb,
    std::size_t n, void* work, std::size_t work_bytes, double* h_nrm2_a,
    double* h_max2_b, double inv_e)
{
    ozaki_compute_scaling_factors_error_bound_cplx(handle, A, lda, m, k, B, ldb,
                                                   n, work, work_bytes,
                                                   h_nrm2_a, h_max2_b, inv_e);
}

template <>
void ozaki_compute_scaling_factors_error_bound<std::complex<double>>(
    cublasHandle_t handle, std::complex<double> const* A, std::size_t lda,
    std::size_t m, std::size_t k, std::complex<double> const* B,
    std::size_t ldb, std::size_t n, void* work, std::size_t work_bytes,
    double* h_nrm2_a, double* h_max2_b, double inv_e)
{
    ozaki_compute_scaling_factors_error_bound_cplx(handle, A, lda, m, k, B, ldb,
                                                   n, work, work_bytes,
                                                   h_nrm2_a, h_max2_b, inv_e);
}

} // namespace gemmul8pp
} // namespace linalg
} // namespace chase

#endif // CHASE_ENABLE_OZAKI
