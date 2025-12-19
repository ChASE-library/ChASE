// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <iostream>
#include <vector>
#include <complex>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>
#include <omp.h>

#include "algorithm/performance.hpp"
#ifdef HAS_CUDA
#include "Impl/chase_gpu/chase_gpu.hpp"
#endif
#include "Impl/chase_cpu/chase_cpu.hpp"
#ifdef USE_NVTX
#include "Impl/chase_gpu/nvtx.hpp"
#endif

using P = double;
using T = std::complex<P>;
using namespace chase;
#ifdef HAS_CUDA
using ARCH = chase::platform::GPU;
#else
using ARCH = chase::platform::CPU;
#endif
using MatrixType = chase::matrix::PseudoHermitianMatrix<T,ARCH>;

int main(int argc, char** argv){

	size_t N = atoi(argv[2]), nev = atoi(argv[3]), nex = atoi(argv[4]); 
    
	auto H  = new MatrixType(N,N);

	#ifdef HAS_CUDA	
		H->allocate_cpu_data();
	#endif

	std::string file_path(argv[1]);

	H->readFromBinaryFile(file_path);

	std::vector<T> V(N*(nev+nex));
	auto Lambda = std::vector<chase::Base<T>>(nev + nex);
	
	#ifdef HAS_CUDA
		auto single = chase::Impl::ChASEGPU<T,MatrixType>(N, nev, nex, H, V.data(), N, Lambda.data());
	#else
		auto single = chase::Impl::ChASECPU<T,MatrixType>(N, nev, nex, H, V.data(), N, Lambda.data());
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

	performanceDecorator.GetPerfData().print();
	
	delete H;
}
