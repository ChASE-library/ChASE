/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include <complex>

namespace chase
{
namespace mpi
{

void blacs_pinfo(int* mypnum, int* nprocs);
void blacs_get(int* icontxt, const int* what, int* val);
void blacs_gridinit(int* icontxt, const char layout, const int* nprow,
                    const int* npcol);
void blacs_gridinfo(int* icontxt, int* nprow, int* npcol, int* myprow,
                    int* mypcol);
std::size_t numroc(std::size_t* n, std::size_t* nb, int* iproc,
                   const int* isrcproc, int* nprocs);
void descinit(std::size_t* desc, std::size_t* m, std::size_t* n,
              std::size_t* mb, std::size_t* nb, const int* irsrc,
              const int* icsrc, int* ictxt, std::size_t* lld, int* info);

template <typename T>
void t_geadd(const char trans, const std::size_t m, std::size_t n, T alpha,
             T* a, const std::size_t ia, const std::size_t ja,
             std::size_t* desc_a, T beta, T* c, const std::size_t ic,
             const std::size_t jc, std::size_t* desc_c);

template <typename T>
void t_ppotrf(const char uplo, const std::size_t n, T* a, const std::size_t ia,
              const std::size_t ja, std::size_t* desc_a);

template <typename T>
void t_psyhegst(const int ibtype, const char uplo, const std::size_t n, T* a,
                const std::size_t ia, const std::size_t ja, std::size_t* desc_a,
                const T* b, const std::size_t ib, const std::size_t jb,
                std::size_t* desc_b, Base<T>* scale);

template <typename T>
void t_ptrtrs(const char uplo, const char trans, const char diag,
              const std::size_t n, const std::size_t nhs, T* a,
              const std::size_t ia, const std::size_t ja, std::size_t* desc_a,
              T* b, const std::size_t ib, const std::size_t jb,
              std::size_t* desc_b);

} // namespace mpi
} // namespace chase

#include "scalapack_templates.inc"
