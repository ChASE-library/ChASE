// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "external/cpu/lapackpp/lapackpp.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

namespace chase
{
namespace tests
{
namespace lanczos
{

template <typename T>
struct IsStdComplex : std::false_type
{
};

template <typename T>
struct IsStdComplex<std::complex<T>> : std::true_type
{
};

template <typename T>
T makeHermitianValue(chase::Base<T> real, chase::Base<T> imag)
{
    if constexpr (IsStdComplex<T>::value)
    {
        return T(real, imag);
    }
    else
    {
        (void)imag;
        return T(real);
    }
}

template <typename T>
T conjugateForHermitian(T value)
{
    if constexpr (IsStdComplex<T>::value)
    {
        return std::conj(value);
    }
    else
    {
        return value;
    }
}

template <typename T>
void initializeReferenceHermitianMatrix(T* matrix, std::size_t n)
{
    using RealT = chase::Base<T>;

    std::fill(matrix, matrix + n * n, T(0));
    for (std::size_t j = 0; j < n; ++j)
    {
        const RealT uj =
            RealT(1.0) + RealT(0.2) * std::cos(RealT(0.07 * j));
        for (std::size_t i = j; i < n; ++i)
        {
            const RealT ui =
                RealT(1.0) + RealT(0.2) * std::cos(RealT(0.07 * i));
            RealT real = RealT(45.0) * ui * uj / RealT(n);
            RealT imag = RealT(0.0);

            if (i == j)
            {
                real += RealT(5.0) + RealT(25.0 * i) / RealT(n - 1);
            }
            else
            {
                const auto distance =
                    static_cast<RealT>(std::min(i - j, std::size_t(16)));
                real += RealT(0.08) * std::cos(RealT(0.11 * (i + j))) /
                        (RealT(1.0) + distance);
                if constexpr (IsStdComplex<T>::value)
                {
                    imag = RealT(0.04) *
                           std::sin(RealT(0.13 * i + 0.17 * j)) /
                           (RealT(1.0) + distance);
                }
            }

            matrix[i + n * j] = makeHermitianValue<T>(real, imag);
            if (i != j)
            {
                matrix[j + n * i] = conjugateForHermitian(matrix[i + n * j]);
            }
        }
    }
}

template <typename T>
void computeReferenceEigenBounds(const T* matrix, std::size_t n,
                                 chase::Base<T>* lambda_min,
                                 chase::Base<T>* lambda_max)
{
    std::vector<T> work(matrix, matrix + n * n);
    std::vector<chase::Base<T>> eigs(n);
    chase::linalg::lapackpp::t_heevd(CblasColMajor, 'N', 'U', n, work.data(),
                                     n, eigs.data());
    *lambda_min = eigs.front();
    *lambda_max = eigs.back();
}

template <typename RealT>
void expectRitzValuesInsideReferenceSpectrum(
    const std::vector<RealT>& ritzv, std::size_t m, std::size_t numvec,
    RealT lambda_min, RealT lambda_max)
{
    const RealT span = std::max(lambda_max - lambda_min, RealT(1.0));
    const RealT slack = RealT(0.05) * span;

    for (std::size_t i = 0; i < numvec; ++i)
    {
        EXPECT_GE(ritzv[i * m], lambda_min - slack);
        EXPECT_LE(ritzv[(i + 1) * m - 1], lambda_max + slack);
    }
}

template <typename RealT>
void expectLanczosUpperBoundNearReference(RealT upperb, RealT lambda_min,
                                          RealT lambda_max)
{
    const RealT span = std::max(lambda_max - lambda_min, RealT(1.0));

    EXPECT_TRUE(std::isfinite(upperb));
    EXPECT_GE(upperb, lambda_max - RealT(0.02) * span);
    EXPECT_LE(upperb, lambda_max + RealT(0.35) * span);
}

} // namespace lanczos
} // namespace tests
} // namespace chase
