/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2018, Simulation Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany
// and
// Copyright (c) 2016-2018, Aachen Institute for Advanced Study in Computational
//   Engineering Science, RWTH Aachen University, Germany All rights reserved.
// License is 3-clause BSD:
// https://github.com/SimLabQuantumMaterials/ChASE/

#pragma once

#include <cstdlib>

#include "algorithm/types.hpp"
#include "chase_mpi_properties.hpp"

//
// This class provides the basic functionality for multi-GPU Hemm.
// The core functionalies are to distribute H, V and W between the GPUs and to
// perform the Hemm operation in distributed manner

namespace chase {
	namespace mpi {

		template <class T>
		class mgpu_cudaHemm {

			public:

				typedef T value_type;

				/* Constructor - sets GPUs, cublas handle, streams */
				mgpu_cudaHemm() {};

				mgpu_cudaHemm(cublasHandle_t handle, cudaStream_t stream, std::size_t m, std::size_t n, std::size_t maxBlock) :
				handle_(handle), stream_(stream), m_(m), n_(n), maxBlock_(maxBlock) {

					// Allocate arrays on GPU
					//cuda_exec(cudaSetDevice(device_id_));
					//cublasSetStream(handle_, stream_);
					cuda_exec(cudaMalloc(&(B_), std::max(n_, m_) * maxBlock_ * sizeof(T)));
					cuda_exec(cudaMalloc(&(IMT_), std::max(n_, m_) * maxBlock_ * sizeof(T)));
					cuda_exec(cudaMalloc(&(H_), n_ * m_  * sizeof(T)));
	
					copied_ = false;

					/* Start timing events */
					cuda_exec(cudaEventCreate(&start));
					cuda_exec(cudaEventCreate(&stop));
					time_copy_H = 0;
				} 

				/* Remove local variabls */
				~mgpu_cudaHemm() { 
					std::cout << "Time required for copying = " << time_copy_H/1e3 << " sec" << std::endl;
					cudaFree(B_);
					cudaFree(IMT_);
					cudaFree(H_);
					cudaEventDestroy(start);
					cudaEventDestroy(stop);	
				}

				/* Divide given matrix H to block and distribute among GPUs */
				void distributeH(T* orig_H) {

					if( !copied_ ) {
						cuda_exec(
							cudaMemcpyAsync(H_, orig_H, m_ * n_ * sizeof(T), cudaMemcpyHostToDevice, stream_));
						copied_ = true;
					}


					cudaEventRecord(start);
					cuda_exec(
						cudaMemcpy(H_, orig_H, n_ * m_ * sizeof(T), cudaMemcpyHostToDevice));
					cudaEventRecord(stop);

					cudaEventSynchronize(stop);
					float local_time;
					cudaEventElapsedTime(&local_time, start, stop);
					time_copy_H += local_time;
					std::cout << "Time = " << local_time/1e3 << " sec. Bandwidth = " << n_*m_*sizeof(T)/(local_time*1e6) << " GB/s" << std::endl;
				}

				/* Divide given tall-skinny matrices V/W into panels and distributed them among GPUs */
				void distributeV();

				/* Compute Hemm */
				void computeHemm(T* buf_init, T* buf_target, std::size_t m, std::size_t n, std::size_t k, std::size_t block, T alpha, T beta, cublasOperation_t transa) {

					cuda_exec(cudaMemcpyAsync(B_, buf_init, block * k * sizeof(T), 
											  cudaMemcpyHostToDevice, stream_));

					cublasTgemm(handle_, transa, CUBLAS_OP_N, m, n, k, &alpha, H_, m_, B_, k,
								&beta, IMT_, m);

					cuda_exec(cudaMemcpyAsync(buf_target, IMT_, m * block * sizeof(T),
							  				  cudaMemcpyDeviceToHost, stream_));
				}

				/* Collect the computed V/W from the GPUs */
				void returnV();

			private:

				std::size_t n_;
				std::size_t m_;
				std::size_t maxBlock_;

				T* B_;
				T* IMT_;
				T* H_;

				bool copied_;

				/// List of GPU devices
				int *gpu_devices = nullptr;
				int device_id_;

  				/// Timing values
  				cudaEvent_t start, stop;
				float time_copy_H;

				/// Cublas handler for
				cublasHandle_t handle_;
			
				/// Cuda streams
				cudaStream_t stream_;

		};
	}  // namespace matrixfree
}  // namespace chase
