#pragma once

// Ozaki-II INT8 constants. GEMMul8 tables extend to N = 20.
// ChASE production range: FP64 [7, 16], FP32 [4, 12].
//
// Moduli and the double-double of P = prod p_i are taken from GEMMul8
// (GEMMul8/src/table.hpp). GEMMul8 stores -P; the table here stores +P.
// rho is not tabulated in GEMMul8; it is sum floor(p_i / 2) on those moduli.

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#ifdef CHASE_ENABLE_OZAKI

namespace chase {
namespace linalg {
namespace gemmul8pp {

inline constexpr unsigned kMinNumModuli          = 2;
inline constexpr unsigned kMinNumModuliChaseFp32 = 4;
inline constexpr unsigned kMinNumModuliChaseFp64 = 7;
inline constexpr unsigned kMaxNumModuliFp32      = 12;
inline constexpr unsigned kMaxNumModuliFp64      = 16;

inline constexpr double kU32 = 0x1p-24; // 2^{-24}
inline constexpr double kU64 = 0x1p-53; // 2^{-53}

inline constexpr std::array<std::int32_t, 20> kModuli = {
    256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
    223, 217, 211, 199, 197, 193, 191, 181, 179, 173};

struct DoubleDouble
{
    double hi;
    double lo;
};

inline constexpr unsigned table_index(unsigned num_moduli)
{
    return num_moduli - kMinNumModuli;
}

// P[i] = prod(p[0], ..., p[i+1]) in double-double. Index = num_moduli - 2.
inline constexpr std::array<DoubleDouble, 19> kP = {{
    { 0x1.fe00000000000p+15,   0x0.0000000000000p+0},  // N = 2
    { 0x1.f806000000000p+23,   0x0.0000000000000p+0},  // N = 3
    { 0x1.ee2de20000000p+31,   0x0.0000000000000p+0},  // N = 4
    { 0x1.dcce450e00000p+39,   0x0.0000000000000p+0},  // N = 5
    { 0x1.c0de2f022e000p+47,   0x0.0000000000000p+0},  // N = 6
    { 0x1.a30f6de308f20p+55,   0x0.0000000000000p+0},  // N = 7
    { 0x1.7d690b03a3244p+63,   0x1.0000000000000p+8},  // N = 8
    { 0x1.552ef6da40ef7p+71,  -0x1.ec00000000000p+14}, // N = 9
    { 0x1.2e88a4e387945p+79,  -0x1.1444000000000p+22}, // N = 10
    { 0x1.078907a2331a3p+87,   0x1.37ac620000000p+31}, // N = 11
    { 0x1.bec64ef0faa26p+94,   0x1.dc188f8900000p+40}, // N = 12
    { 0x1.703d73109e93ep+102, -0x1.2f97c1b215000p+48}, // N = 13
    { 0x1.1e3fc471eb44fp+110, -0x1.1ff7bc8b72980p+53}, // N = 14
    { 0x1.b88e245754182p+117, -0x1.df666905d3cbcp+63}, // N = 15
    { 0x1.4c232965d6663p+125, -0x1.616c352d64acap+71}, // N = 16
    { 0x1.ef9c77c5f5ec7p+132,  0x1.b141114c878cep+77}, // N = 17
    { 0x1.5e69a0aef6e03p+140, -0x1.35acfec4e4296p+85}, // N = 18
    { 0x1.ea07b6b4ad3d8p+147, -0x1.087f623ab88f0p+89}, // N = 19
    { 0x1.4b27367819129p+155, -0x1.595f0ab0d75c5p+98}  // N = 20
}};

// rho[i] = sum_{j=0}^{i+1} floor(p[j] / 2). Index = num_moduli - 2.
inline constexpr std::array<std::uint16_t, 19> kRho = {
    255, 381, 506, 629, 749, 868, 984, 1098, 1211, 1322,
    1430, 1535, 1634, 1732, 1828, 1923, 2013, 2102, 2188};

inline constexpr std::array<int, 19> kCeilLog2Rho = {
    8, 9, 9, 10, 10, 10, 10, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 12, 12};

// t = 1 / sqrt(32 (P - 1)), correctly rounded to binary64 from the exact P.
inline constexpr std::array<double, 19> kT = {
    0x1.6ac029a6bd3f7p-11, 0x1.6ce467d19fdd1p-15, 0x1.7082379d9a471p-19,
    0x1.77298d6440747p-23, 0x1.82a948a27e336p-27, 0x1.902d363542cbcp-31,
    0x1.a3768b22ce13bp-35, 0x1.bb809c8304ca0p-39, 0x1.d6fb0879692f0p-43,
    0x1.f8a0897da132ep-47, 0x1.120cf14261d56p-50, 0x1.2ddcd97cb8602p-54,
    0x1.56602b838873fp-58, 0x1.864ac6d9dcc64p-62, 0x1.c1805e3a0a1b4p-66,
    0x1.0432c13089422p-69, 0x1.357236e38f2eap-73, 0x1.7210c1eb42727p-77,
    0x1.c22b3c99c8a19p-81};

// r_32 = (1 + u_32)(N + 2) u_64 rho P + (1/2) u_32 P
inline constexpr std::array<double, 19> kR32 = {
    0x1.fe007f00807f0p-10, 0x1.f806ea6a4b4a7p-2,  0x1.ee2f504a83b1dp+6,
    0x1.dcd04598fcdb8p+14, 0x1.c0e0bfa738555p+22, 0x1.a3128d2139658p+30,
    0x1.7d6c9f4b002e4p+38, 0x1.5532e4e921045p+46, 0x1.2e8cd63bd6c5ep+54,
    0x1.078d59600a936p+62, 0x1.beced6a3c8817p+69, 0x1.70458911144b8p+77,
    0x1.1e46e784f6472p+85, 0x1.b89a83432933cp+92, 0x1.4c2d958656e0ep+100,
    0x1.efadbcb52f573p+107, 0x1.5e7714ecbdf43p+115, 0x1.ea1c57b0a2fcep+122,
    0x1.4b366a2a7be4ap+130};

// r_64 = (1 + 3 u_64) 2^{1 + ceil(log2 rho)} (N + 2) u_64^2 rho P + (3/2) u_64 P
inline constexpr std::array<double, 19> kR64 = {
    0x1.7e8000003f804p-37, 0x1.7a048000ea6a5p-29, 0x1.72a269816e4a8p-21,
    0x1.659ab3ce8115fp-13, 0x1.50a6a346c3ca1p-5,  0x1.3a4b92708531ep+3,
    0x1.1e0ec849e2e9ep+11, 0x1.ffc67266d1de0p+18, 0x1.c5ccf776d620dp+26,
    0x1.8b4d8b95da95ep+34, 0x1.4f14bb56dac4ep+42, 0x1.142e166ccef0ap+50,
    0x1.ad5fa6e3f97f8p+57, 0x1.4a6a9b72fac13p+65, 0x1.f234be6c229cfp+72,
    0x1.73b55a198c2dfp+80, 0x1.06cf38b90a1f1p+88, 0x1.6f85c9ac89cd3p+95,
    0x1.f0bad2a760c11p+102};

inline constexpr DoubleDouble P_dd(unsigned num_moduli)
{
    return kP[table_index(num_moduli)];
}

inline constexpr double P(unsigned num_moduli)
{
    DoubleDouble const dd = P_dd(num_moduli);
    return dd.hi + dd.lo;
}

inline constexpr std::uint16_t rho(unsigned num_moduli)
{
    return kRho[table_index(num_moduli)];
}

inline constexpr int ceil_log2_rho(unsigned num_moduli)
{
    return kCeilLog2Rho[table_index(num_moduli)];
}

inline constexpr double t(unsigned num_moduli)
{
    return kT[table_index(num_moduli)];
}

inline constexpr double r32(unsigned num_moduli)
{
    return kR32[table_index(num_moduli)];
}

inline constexpr double r64(unsigned num_moduli)
{
    return kR64[table_index(num_moduli)];
}

template <typename T>
inline constexpr bool is_fp32_v = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;

template <typename T>
inline constexpr double r_b(unsigned num_moduli)
{
    if constexpr (is_fp32_v<T>)
    {
        return r32(num_moduli);
    }
    else
    {
        return r64(num_moduli);
    }
}

template <typename T>
inline constexpr unsigned kMaxNumModuli =
    is_fp32_v<T> ? kMaxNumModuliFp32 : kMaxNumModuliFp64;

template <typename T>
inline constexpr unsigned kMinNumModuliChase =
    is_fp32_v<T> ? kMinNumModuliChaseFp32 : kMinNumModuliChaseFp64;

inline double two_pow_floor_log2(double max_abs)
{
    if (!(max_abs > 0.0))
    {
        return 0.0;
    }
    return std::exp2(std::floor(std::log2(max_abs)));
}

// γ̃_n = (n u) / (1 - n u).
inline double gamma_n(std::size_t n, double u)
{
    double const nu = static_cast<double>(n) * u;
    return nu / (1.0 - nu);
}

// IEEE GEMM residual in the (1,2)-norm, unit columns of V (FP64):
//   ||E^f||_{1,2} <= γ_n ||Ā||_∞  (caller may pass ρ(Ā) or ||Ā||_∞).
inline double ieee_gemm_error_bound_12(double rho_A, std::size_t n)
{
    return gamma_n(n, kU64) * rho_A;
}

// Ozaki-II GEMM residual in the (1,2)-norm (unit columns of V):
//   ||E||_{1,2} <= t √n ||Ā||_∞ max 2^{β'}
//                + t √n ||2^{α'}||_2
//                + t² (n + r) ||2^{α'}||_2 max 2^{β'}
// Primes from ozaki_compute_scaling_factors_error_bound.
// rho_A: spectral radius or infinity-norm surrogate for Ā.
template <typename T>
inline double ozaki_gemm_error_bound(double rho_A, double nrm2_two_pow_a,
                                     double max_two_pow_b, std::size_t n,
                                     unsigned num_moduli)
{
    unsigned s = num_moduli;
    if (s < kMinNumModuli)
    {
        s = kMinNumModuli;
    }
    if (s > kMaxNumModuli<T>)
    {
        s = kMaxNumModuli<T>;
    }

    double const t_n    = t(s);
    double const r      = r_b<T>(s);
    double const n_d    = static_cast<double>(n);
    double const sqrt_n = std::sqrt(n_d);
    double const a      = nrm2_two_pow_a;
    double const b      = max_two_pow_b;

    return t_n * sqrt_n * rho_A * b + t_n * sqrt_n * a +
           t_n * t_n * (n_d + r) * a * b;
}

// Legacy max-norm cheap surrogate (Lemma 5); kept for pchase / comparison.
// https://arxiv.org/pdf/2602.02549
template <typename T>
inline double local_gemm_error_bound(double max_abs_A, double max_abs_B,
                                     double inf_norm_A, double one_norm_B,
                                     std::size_t k, unsigned num_moduli)
{
    unsigned s = num_moduli;
    if (s < kMinNumModuli)
    {
        s = kMinNumModuli;
    }
    if (s > kMaxNumModuli<T>)
    {
        s = kMaxNumModuli<T>;
    }

    double const t_n   = t(s);
    double const r     = r_b<T>(s);
    double const k_d   = static_cast<double>(k);
    double const slice = 64.0 * std::sqrt(k_d);
    double const a     = two_pow_floor_log2(max_abs_A) * slice;
    double const b     = two_pow_floor_log2(max_abs_B) * slice;

    return t_n * (inf_norm_A * b + a * one_norm_B + (k_d + r) * t_n * a * b);
}

// U2-model sum. If ||E_j^f|| <= ||E_0^f|| for all j, then
//   ||E_{i+1}^p|| <= ||E_0^f|| * u2_model_sum(degree, lambda_1, lower, upper)
// with i = degree - 1,
//   sum_j w_j |eta_j U_{i-j}((lambda_1 - c) / e) / eta_{i+1}|,
//   w_0 = 1, w_j = 2 for j >= 1,
//   c = (upper + lower) / 2, e = (upper - lower) / 2,
//   eta_0 = 1, eta_t = eta_{t-1} / sigma_t, sigma_1 = e / (lambda_1 - c).
inline double u2_model_sum(std::size_t degree, double lambda_1, double lower,
                           double upper)
{
    if (degree == 0)
    {
        return 0.0;
    }

    double const c = 0.5 * (upper + lower);
    double const e = 0.5 * (upper - lower);
    double const x = (lambda_1 - c) / e;
    double const sigma_1 = e / (lambda_1 - c);

    std::vector<double> nu(degree, 0.0);
    std::vector<double> eta(degree + 1, 0.0);
    std::vector<double> sigma(degree + 1, 0.0);

    eta[0] = 1.0;
    sigma[1] = sigma_1;
    eta[1] = eta[0] / sigma[1];

    for (std::size_t t = 2; t <= degree; ++t)
    {
        sigma[t] = 1.0 / (2.0 / sigma_1 - sigma[t - 1]);
        eta[t] = eta[t - 1] / sigma[t];
    }

    double const eta_ip1 = eta[degree];

    nu[0] = 1.0;
    if (degree >= 2)
    {
        nu[1] = 2.0 * x;
        for (std::size_t k = 2; k <= degree - 1; ++k)
        {
            nu[k] = 2.0 * x * nu[k - 1] - nu[k - 2];
        }
    }

    std::size_t const i = degree - 1;
    double sum = 0.0;
    for (std::size_t j = 0; j <= i; ++j)
    {
        double const w = (j == 0) ? 1.0 : 2.0;
        sum += w * std::abs(eta[j] * nu[i - j] / eta_ip1);
    }
    
    return sum;
}

} // namespace gemmul8pp
} // namespace linalg
} // namespace chase

#endif // CHASE_ENABLE_OZAKI
