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
using MatrixType = chase::matrix::QuasiHermitianMatrix<T,ARCH>;
//using MatrixType = chase::matrix::Matrix<T,ARCH>;

P data_norm(T * data, size_t m, size_t n){
	P norm = 0.0;
	for(auto i = 0; i < m; i++){
		for(auto j = 0; j < n; j++){
			norm += std::norm(data[i*n + j]);
		}
	}
	return norm;
}

void subtract(T * data1, T* data2, size_t m, size_t n){
	for(auto i = 0; i < m; i++){
		for(auto j = 0; j < n; j++){
			data1[i*n +j] = data1[i*n + j] - data2[i*n + j];
		}
	}
}

int main(int argc, char** argv){

	#ifdef HAS_CUDA
		std::cout << "================= ChASE GPU mode ====================" << std::endl;
	#else
		std::cout << "================= ChASE CPU mode ====================" << std::endl;
	#endif

	std::cout << "\n-----------------------------------------------------" << std::endl;
	std::cout << "                 Reading matrix files                  " << std::endl;
	std::cout << "-----------------------------------------------------\n" << std::endl;

	// ============= ChASE READ Matrix ============ //
	
	size_t k = 5;
	//size_t k = 1472;
	size_t N = 2*k, nev = 3, nex = 2; 
    
	auto H  = new MatrixType(N,N);

	#ifdef HAS_CUDA	
		H->allocate_cpu_data();
	#endif

	//H->readFromBinaryFile("../../../Codes/chase-programs/data/cdouble_random_BSE.bin");
	H->readFromBinaryFile("../../../Codes/chase-programs/data/cdouble_tiny_random_BSE.bin");
	//H->readFromBinaryFile("../data/2x2x2_Silicon_QuasiHermitian.bin");
	
	#ifdef HAS_CUDA	
		std::cout << "Second element of first column " << H->cpu_data()[1] << std::endl;
	#else
		std::cout << "Second element of first column " << H->data()[1] << std::endl;
	#endif
/*
	auto eigsl_H  = new chase::matrix::Matrix<T,ARCH>(N,1);
	eigsl_H->readFromBinaryFile("../data/eigs_cdouble_random_BSE.bin");

	std::cout << "Eigenvalues are : " << std::endl;
	for(auto i = 0; i < nev; i++){
		std::cout << std::real(eigsl_H->data()[i]) << " ";
	}
*/
	//chase::linalg::internal::cpu::flipLowerHalfMatrixSign(N,N,H->data(),N);

	std::cout << std::endl;

	auto V  = chase::matrix::Matrix<T,ARCH>(N,nev+nex);
	auto HV = chase::matrix::Matrix<T,ARCH>(N,nev+nex);
	auto Lambda = std::vector<chase::Base<T>>(nev + nex);
	
	#ifdef HAS_CUDA
    		V.allocate_cpu_data();
		auto single = chase::Impl::ChASEGPU<T,MatrixType>(N, nev, nex, H, V.data(), V.ld(), Lambda.data());
	#else
		auto single = chase::Impl::ChASECPU<T,MatrixType>(N, nev, nex, H, V.data(), V.ld(), Lambda.data());
	#endif	

	single.initVecs(true);

	std::cout << "\n-----------------------------------------------------" << std::endl;
	std::cout << "                      Matrix info                      " << std::endl;
	std::cout << "-----------------------------------------------------\n" << std::endl;

	std::cout << "Matrix size                         = " << H->ld() << std::endl;
	bool matrix_sym = single.checkSymmetryEasy();
	std::cout << "Is Matrix Symmetric                 ? " << matrix_sym << std::endl;
	bool matrix_pse = single.checkPseudoHermicityEasy();
	std::cout << "Is Matrix Pseudo-Hermitian          ? " << matrix_pse << std::endl;
	
	std::cout << "\n" << std::endl;
	
	std::cout << "\n-----------------------------------------------------" << std::endl;
	std::cout << "                     Chase Solver                      " << std::endl;
	std::cout << "-----------------------------------------------------\n" << std::endl;

    	auto& config = single.GetConfig();
    	//Tolerance for Eigenpair convergence
    	config.SetTol(1e-10);
    	//Initial filtering degree
    	config.SetDeg(20);
    	//Optimi(S)e degree
    	config.SetOpt(true);
	//ChASE Max iter
        config.SetMaxIter(50);
	config.SetNumLanczos(4);
	//config.SetDecayingRate(0.90);

	PerformanceDecoratorChase<T> performanceDecorator(&single);

	chase::Solve(&performanceDecorator);

        //Output

	performanceDecorator.GetPerfData().print();
           
        Base<T>* resid = single.GetResid();
        std::cout << "Finished Problem #1"
        << "\n";
	//std::cout << "Printing first 5 eigenvalues and residuals\n";
        std::cout
        << "| Index |       Eigenvalue      |         Residual      |\n"
        << "|-------|-----------------------|-----------------------|\n";
        std::size_t width = 20;
        std::cout << std::setprecision(12);
        std::cout << std::setfill(' ');
        std::cout << std::scientific;
        std::cout << std::right;
        for (auto i = 0; i < std::min(std::size_t(nev+nex), nev+nex); ++i)
            std::cout << "|  " << std::setw(4) << i + 1 << " | "
                    << std::setw(width) << Lambda[i] << "  | "
                    << std::setw(width) << resid[i] << "  |\n";
        std::cout << "\n\n\n";

	std::cout << "\n" << std::endl;
	delete H;
}
