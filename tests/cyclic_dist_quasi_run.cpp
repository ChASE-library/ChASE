// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <complex>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "algorithm/performance.hpp"
#ifdef HAS_CUDA
#include "Impl/pchase_gpu/pchase_gpu.hpp"
#else
#include "Impl/pchase_cpu/pchase_cpu.hpp"
#endif

using T = std::complex<double>;
using namespace chase;

#ifdef HAS_CUDA
using ARCH = chase::platform::GPU;
using BackendType = chase::grid::backend::NCCL;
#else
using ARCH = chase::platform::CPU;
#endif

int main(int argc, char** argv){
        
	MPI_Init(&argc, &argv);

	size_t N = atoi(argv[2]), nev = atoi(argv[3]), nex = atoi(argv[4]), mb = 64;
    
	int world_rank,world_size;
    
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    	int dims_[2];
    	dims_[0] = dims_[1] = 0;

    	MPI_Dims_create(world_size, 2, dims_);

    	std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            dims_[0], dims_[1], MPI_COMM_WORLD);

    
	auto Hmat = chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, ARCH>(N, N, mb, mb, mpi_grid);

	#ifdef HAS_CUDA
    		Hmat.allocate_cpu_data();
    	#endif
        	
	std::string file_path(argv[1]);

        Hmat.readFromBinaryFile(file_path);

	auto Lambda = std::vector<chase::Base<T>>(nev + nex);

        auto Vec = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, ARCH>(N, nev + nex, mb, mpi_grid);

	#ifdef HAS_CUDA
    		Hmat.allocate_cpu_data();
   		auto single = chase::Impl::pChASEGPU<decltype(Hmat), decltype(Vec), BackendType>(nev, nex, &Hmat, &Vec, Lambda.data());
	#else
    		auto single = chase::Impl::pChASECPU(nev, nex, &Hmat, &Vec, Lambda.data());
	#endif
    	
	auto& config = single.GetConfig();
    	config.SetTol(1e-10);
    	config.SetDeg(atoi(argv[5]));
	if(argv[6][0] == 'Y'){
    		config.SetOpt(true);
	}else{
    		config.SetOpt(false);
	}
        config.SetMaxIter(50);

	PerformanceDecoratorChase<T> performanceDecorator(&single);

	chase::Solve(&performanceDecorator);

	if(world_rank == 0){
		performanceDecorator.GetPerfData().print();
	}
}
