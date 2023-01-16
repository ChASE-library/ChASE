/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <complex>
#include <memory>
#include <random>
#include <vector>
#include <iostream>
#include <type_traits>
#include <cstdlib>

#include "algorithm/types.hpp"
#include "ChASE-MPI/blas_templates.hpp"

using T = std::complex<double>;

using namespace chase;
using namespace chase::mpi;

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
MPI_Datatype getMPI_Type<std::complex<float>>()
{
    return MPI_COMPLEX;
}

template <>
MPI_Datatype getMPI_Type<std::complex<double>>()
{
    return MPI_DOUBLE_COMPLEX;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    T zero = T(0.0);
    T one = T(1.0);

    int M;
    int n;
    int m, mb;
    Base<T> cond;
    int len;
    int info;

    if(argc < 3){
        if(rank == 0){
	    std::cout << "Not enough command line arguments are provided.\n";
    	    std::cout << "Run with default matrix size..." << std::endl; 	    
	}
	M = 1000;
	n = 500;
    }else{
        M = atoi(argv[1]);
	n = atoi(argv[2]);
    }

    if(M % size == 0){
        len = M / size;
    }else{
    	len = std::min(M, M / size + 1);
    }

    if(rank < size - 1){
        m = len;
    }else{
        m = M - (size - 1) * len;
    }

    //std::default_random_engine gen(1231.0 * rank);
    std::mt19937 gen(1337.0);
    //std::mt19937 gen(1337.0 * rank);
    std::normal_distribution<> d;
    
    mb = m;
    if (rank == size - 1 && size != 1)
    {
        mb = (M - m) / (size - 1);
    }
    
    auto V = std::vector<T>(m * n);
    auto A = std::vector<T>(n * n);
    auto A_b = std::vector<T>(n * n);
    auto ev = std::vector<Base<T>>(n);
/*    for(auto j = 0; j < m * n; j++){
	auto rnd = getRandomT<T>([&]() { return d(gen); });    
        V[j] = rnd;
    }    
 */   
    for (auto j = 0; j < n; j++)
    {
        std::size_t cnt = 0;
        for (auto i = 0; i < M; i++)
        {
            auto rnk = (i / mb) % size;
            auto rnd = getRandomT<T>([&]() { return d(gen); });
            if (rank == rnk)
            {
                V[cnt + j * m] = rnd;
                cnt++;
            }
        }
    }

    //syherk: V' * V -> A
    t_syherk('U', 'C', n, m, &one, V.data(), m, &zero, A.data(), n);
    MPI_Allreduce(MPI_IN_PLACE, A.data(), n * n, getMPI_Type<T>(), MPI_SUM, MPI_COMM_WORLD);
    A_b = A;
    t_heevd(LAPACK_COL_MAJOR, 'N', 'L', n, A_b.data(), n, ev.data()); 
    cond = std::sqrt(ev[n-1]/ev[0]);

    if(rank == 0){
        std::cout << "M: " << M << ", N: " << n << ", rcond: " << cond ;
    }

    //Cholesky QR on V
    info = t_potrf('U', n, A.data(), n);
    if (info == 0){
        t_trsm('R', 'U', 'N', 'N', m, n, &one, A.data(), n, V.data(), m);
    }    

    //syherk: V' * V -> A
    t_syherk('U', 'C', n, m, &one, V.data(), m, &zero, A.data(), n);
    MPI_Allreduce(MPI_IN_PLACE, A.data(), n * n, getMPI_Type<T>(), MPI_SUM, MPI_COMM_WORLD);
    A_b = A;
    t_heevd(LAPACK_COL_MAJOR, 'N', 'L', n, A_b.data(), n, ev.data());
    cond = std::sqrt(ev[n-1]/ev[0]);

    if(rank == 0){
        std::cout << ", cond: " << cond << std::endl;
    }

    MPI_Finalize();
}

