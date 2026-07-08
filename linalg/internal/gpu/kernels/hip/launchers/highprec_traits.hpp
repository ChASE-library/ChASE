#pragma once

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

// Compile-time switch for panel high-precision protection.
// Default: off (keeps current behavior).
// Set -DCHASE_PANEL_HIPREC=1 in your build to enable QD/DP protected kernels.
#ifndef CHASE_PANEL_HIPREC
#define CHASE_PANEL_HIPREC 0
#endif

namespace chase
{
namespace linalg
{
namespace internal
{
namespace hip
{

// Double-double (QD) scalar: x ~= hi + lo with ~31 extra bits of precision.
struct QD
{
    double hi;
    double lo;

    __host__ __device__ __forceinline__ QD(double h = 0.0, double l = 0.0)
        : hi(h), lo(l)
    {
    }
};

struct ComplexQD
{
    QD re;
    QD im;

    __host__ __device__ __forceinline__ ComplexQD(QD r = QD{}, QD i = QD{})
        : re(r), im(i)
    {
    }
};

// Error-free transformations (EFT)
__device__ __forceinline__ QD two_sum(double a, double b)
{
    const double s = a + b;
    const double v = s - a;
    const double e = (a - (s - v)) + (b - v);
    return QD{s, e};
}

__device__ __forceinline__ QD two_prod(double a, double b)
{
    const double p = a * b;
    const double e = fma(a, b, -p);
    return QD{p, e};
}

// Sloppy but stable double-double addition with renormalization.
__device__ __forceinline__ QD qd_add_qd(QD a, QD b)
{
    QD s = two_sum(a.hi, b.hi);
    const double lo_sum = s.lo + (a.lo + b.lo);
    s = two_sum(s.hi, lo_sum);
    return s;
}

__device__ __forceinline__ QD qd_neg(QD x) { return QD{-x.hi, -x.lo}; }

__device__ __forceinline__ QD qd_sub_qd(QD a, QD b) { return qd_add_qd(a, qd_neg(b)); }

// Multiply QD by a double scalar: (x_hi + x_lo) * a
__device__ __forceinline__ QD qd_mul_double(QD x, double a)
{
    QD p1 = two_prod(a, x.hi);
    QD p2 = two_prod(a, x.lo);
    return qd_add_qd(p1, p2);
}

__device__ __forceinline__ double qd_to_double(QD x) { return x.hi + x.lo; }

// Complex QD arithmetic where inputs are plain hipDoubleComplex (double precision),
// but the result accumulates in QD to reduce cancellation in panel T-matrix build.
__device__ __forceinline__ ComplexQD complex_mul_double_as_qd(hipDoubleComplex a,
                                                                 hipDoubleComplex b)
{
    // (ar + i*ai) * (br + i*bi) = (ar*br - ai*bi) + i*(ar*bi + ai*br)
    const double ar = hipCreal(a);
    const double ai = hipCimag(a);
    const double br = hipCreal(b);
    const double bi = hipCimag(b);

    QD ac = two_prod(ar, br);
    QD bd = two_prod(ai, bi);
    QD ad = two_prod(ar, bi);
    QD bc = two_prod(ai, br);

    ComplexQD out;
    out.re = qd_sub_qd(ac, bd);
    out.im = qd_add_qd(ad, bc);
    return out;
}

// Multiply hipDoubleComplex (plain double parts) by ComplexQD.
__device__ __forceinline__ ComplexQD complex_mul_doublecomplex_by_qd(hipDoubleComplex a,
                                                                        ComplexQD b)
{
    const double ar = hipCreal(a);
    const double ai = hipCimag(a);

    // real = ar*b.re - ai*b.im
    // imag = ar*b.im + ai*b.re
    QD real_part = qd_sub_qd(qd_mul_double(b.re, ar), qd_mul_double(b.im, ai));
    QD imag_part = qd_add_qd(qd_mul_double(b.im, ar), qd_mul_double(b.re, ai));
    return ComplexQD(real_part, imag_part);
}

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase

