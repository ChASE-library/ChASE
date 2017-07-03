/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#ifndef BMMMPI_H
#define BMMMPI_H

#include "template_wrapper.hpp"
#include <complex>
#include <cstring> //mempcpy
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mkl.h>
#include <mpi.h>
#include <string>

#ifdef GPU_MODE
#include <MMMGPU.hpp>
#endif

//TODO this should be a class
// then we don't need template variables

typedef int CHASE_MPIINT;
#define MKL_Complex16 std::complex<double>

template <typename T>
MPI_Datatype getMPI_Type();

template <>
MPI_Datatype getMPI_Type<float>()
{
    return MPI_FLOAT;
}

template <>
MPI_Datatype getMPI_Type<double>()
{
    return MPI_DOUBLE;
}

template <>
MPI_Datatype getMPI_Type<std::complex<float> >()
{
    return MPI_COMPLEX;
}

template <>
MPI_Datatype getMPI_Type<std::complex<double> >()
{
    return MPI_DOUBLE_COMPLEX;
}

template <typename T>
struct MPI_Handler {
    MPI_Group ROW, COL, origGroup;
    MPI_Comm ROW_COMM, COL_COMM;
    MPI_Comm CART_COMM;
    CHASE_MPIINT* ranks_row;
    CHASE_MPIINT* ranks_col;
    CHASE_MPIINT dims[2];
    CHASE_MPIINT coord[2];
    CHASE_INT off[2];
    T* A;
    T* B;
    T* C;
    T* IMT;
    CHASE_INT global_n, m, n, nev;
    CHASE_MPIINT nprocs, rank;
    char next;
    CHASE_INT initialized;
#ifdef GPU_MODE
  ChaseGpuHelper<T> *chase_gpu_helper;
#endif
};

template <typename T>
void MPI_handler_init(MPI_Handler<T>* MPI_hand, MPI_Comm comm,
    CHASE_INT global_n, CHASE_INT nev);

template <typename T>
void MPI_destroy(MPI_Handler<T>* MPI_hand);

template <typename T>
void MPI_load(MPI_Handler<T>* MPI_hand);

template <typename T>
void MPI_distribute_H(MPI_Handler<T>* MPI_Hand, T* H_Full);

template <typename T>
void MPI_distribute_V(MPI_Handler<T>* MPI_Hand, T* V, CHASE_INT nev);

template <typename T>
void MPI_doGemm(MPI_Handler<T>* MPI_hand, T alpha, T beta, CHASE_INT offset,
    CHASE_INT nev);

template <typename T>
void MPI_get_off(MPI_Handler<T>* MPI_hand, CHASE_INT* xoff, CHASE_INT* yoff,
    CHASE_INT* xlen, CHASE_INT* ylen);

template <typename T>
void MPI_get_C(MPI_Handler<T>* MPI_hand, CHASE_INT* COff, CHASE_INT* CLen, T* C,
    CHASE_INT nev);

template <typename T>
void shiftA(MPI_Handler<T>* MPI_Hand, T c);

template <typename T>
void MPI_lock_vectors(MPI_Handler<T>* MPI_hand, CHASE_INT nev);

template <typename T>
void cpy_vectors(MPI_Handler<T>* MPI_hand, CHASE_INT new_converged,
    CHASE_INT locked);

template <typename T>
void debug_H(MPI_Handler<T>* MPI_Hand);
template <typename T>
void debug_IMT(MPI_Handler<T>* MPI_hand);
template <typename T>
void debug_B(MPI_Handler<T>* MPI_hand);
template <typename T>
void debug_C(MPI_Handler<T>* MPI_hand);
template <typename T>
void debug_A(MPI_Handler<T>* MPI_hand);

#include "BMMMPI_impl.hpp"

#endif
