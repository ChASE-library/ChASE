/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstdlib>
#include <iostream>
#include <cublas_v2.h>

#include <chrono>

#include "algorithm/types.hpp"
#include "ChASE-MPI/chase_mpi_properties.hpp"
#include "ChASE-MPI/blas_cuda_wrapper.hpp"

/**
 * This class provides the basic functionality for a series of  a multi-GPU Hemm-s.
 * The core functionalies are to distribute H, V and W between the GPU devices and to
 * perform the Hemm operation in distributed manner among all GPU devices. 
 * The class supports two type of operations
 *
 * 		W = alpha * H * V + beta * W (refered as 'cAb' operation)
 * or
 * 		V = alpha * H^C * W + beta * V (refered as 'bAc' operation)
 *
 * 	such that not additional redististribution of H is required between successive Hemm calls.
 */

using namespace std::chrono;

namespace chase {
	namespace mpi {
		//! A class to defines the multi-GPU implementation of DLA within node.
		/*! This class mainly implements a multi-GPU for `HEMM`, which is able to be executed either with 1 MPI managing multi-GPUs or each GPU being bounded to one MPI rank.
			This class mainly provides the basic functionality for a series of  a multi-GPU Hemm-s.
			The core functionalies are to distribute H, V and W between the GPU devices and to
			perform the Hemm operation in distributed manner among all GPU devices. 
			The class supports two type of operations
			  - W = alpha * H * V + beta * W (refered as 'cAb' operation)
			  - V = alpha * H^C * W + beta * V (refered as 'bAc' operation)

			such that not additional redististribution of H is required between successive Hemm calls. 
			
			This class also provides single-GPU implementation of `gegqr` and `RR_kernel`.
		  
		*/
		template <class T>
		class mgpu_cudaDLA {

			public:

				typedef T value_type;

				/* Constructor - sets GPUs, allocate device arrays, create cublas handles and streams */
				mgpu_cudaDLA() {};

				/* Constructor - sets GPUs, allocate device arrays, create cublas handles and streams */
				//! A constructor of mgpu_cudaDLA which sets GPUs, allocate device arrays, create cublas and cusolver handles and streams.
				/*!
					@param matrix_properties: 	it is an object of ChaseMpiProperties, which defines the MPI environment and data distribution scheme in ChASE-MPI.
					@param m: row number of local matrix stored on each MPI rank.
					@param n: column number of local matrix stored on each MPI rank.
					@param maxBlock: maximum number of column of the rectangular matrix `V`, which equals to `nev+nex`.
				*/
				mgpu_cudaDLA(ChaseMpiProperties<T>* matrix_properties, 
					      std::size_t m, 
					      std::size_t n, 
					      std::size_t maxBlock) :
					      m_(m), n_(n), maxBlock_(maxBlock) {
				
					N_ = matrix_properties->get_N();
        			nev_ = matrix_properties->GetNev();	
        			nex_ = matrix_properties->GetNex();	

    				matrix_properties->get_offs_lens(r_offs_, r_lens_, r_offs_l_, c_offs_, c_lens_, c_offs_l_);
    				mb_ = matrix_properties->get_mb();
    				nb_ = matrix_properties->get_nb();

    				mblocks_ = matrix_properties->get_mblocks();
    				nblocks_ = matrix_properties->get_nblocks();

					MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
					MPI_Comm_size(shmcomm, &shmsize_);
					MPI_Comm_rank(shmcomm, &shmrank_);
					MPI_Comm_rank(MPI_COMM_WORLD, &globalrank_);
	
					/* Get number of available GPU devices */
					cuda_exec(cudaGetDeviceCount(&num_devices));

					/* Divide devices among MPI ranks */
					num_devices_per_rank = num_devices / shmsize_;

					/* Allocate array for device indices, handles and streams */
					handle_ = (cublasHandle_t*) malloc(num_devices_per_rank * sizeof(cublasHandle_t));
					stream_ = (cudaStream_t*) malloc(num_devices_per_rank * sizeof(cudaStream_t));

					cusolverH_ = (cusolverDnHandle_t*) malloc(num_devices_per_rank * sizeof(cusolverDnHandle_t));

                                        stream2_ = (cudaStream_t*) malloc(num_devices_per_rank * sizeof(cudaStream_t));

					/* Populate list of devices, create handles and streams for each device */
					for (int dev=0; dev<num_devices_per_rank; dev++) {
						cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + dev));
						cublasCreate(&handle_[dev]);
						cuda_exec(cudaStreamCreate(&stream_[dev]));
						cublasSetStream(handle_[dev], stream_[dev]);
						cusolverDnCreate(&cusolverH_[dev]);
                                                cuda_exec(cudaStreamCreate(&stream2_[dev]));			
                                                cusolverDnSetStream(cusolverH_[dev], stream2_[dev]);						
					}

					/* Allocate arrays to hold pointers to memory allocations on each device for
					 * matrices H, B and IMT */
					H_ = (T**) malloc(num_devices_per_rank * sizeof(T*));
					B_ = (T**) malloc(num_devices_per_rank * sizeof(T*));
					IMT_ = (T**) malloc(num_devices_per_rank * sizeof(T*));
					WRKSPACE_ = (T**) malloc(num_devices_per_rank * sizeof(T*));

					/* Pointers to memory location of the HEMM operands. The class always computes W = H * V + W operation */
					W = (T**) malloc(num_devices_per_rank * sizeof(T*));
					V = (T**) malloc(num_devices_per_rank * sizeof(T*));
					
					/* Allocate arrays to hold pitch values for each 2D array */
					pitchB = (std::size_t*) malloc(num_devices_per_rank * sizeof(std::size_t));
					pitchH = (std::size_t*) malloc(num_devices_per_rank * sizeof(std::size_t));
					pitchIMT = (std::size_t*) malloc(num_devices_per_rank * sizeof(std::size_t));
					pitchWRK = (std::size_t*) malloc(num_devices_per_rank * sizeof(std::size_t));

					/* Pointers to pitch values of the HEMM operands */
					pitchW = (std::size_t*) malloc(num_devices_per_rank * sizeof(std::size_t));
					pitchV = (std::size_t*) malloc(num_devices_per_rank * sizeof(std::size_t));

					/* Compute number of tiles of matrix H
					 * The number of tiles depends on number of the available GPU devices */
					ntile_n_ = sqrt(num_devices_per_rank);
					ntile_m_ = num_devices_per_rank/ntile_n_;

					/* Compute tile dimensions */
					dim_tile_n_ = std::min(n_, (n_+ntile_n_-1)/ntile_n_);
					dim_tile_m_ = std::min(m_, (m_+ntile_m_-1)/ntile_m_);
					
					/* Set leading dimensions of the GPU arrays */
					/// TODO: ldB and ldIMT does not to be large as ldWRK but rather dim_tile_m and dim_tile_n_, respectively
					ldWRK = std::max(dim_tile_m_, dim_tile_n_);
					ldB = ldIMT = ldWRK;
					ldH = dim_tile_m_;
#ifdef MGPU_TIMER
					std::cout << "[MGPU_HEMM] MPI rank global/local = " << globalrank_ << "/" << shmrank_ << std::endl;
					std::cout << "[MGPU_HEMM] GPUs per rank   = " << num_devices_per_rank << std::endl;
					std::cout << "[MGPU_HEMM] Number of tiles = "<<ntile_m_ << " x " << ntile_n_ << std::endl;
					std::cout << "[MGPU_HEMM] Tile dimension  = "<<dim_tile_m_ << " x " << dim_tile_n_ << std::endl;
#endif
					int tile_x = 0;
					int tile_y = 0;

					/* Pass through the tiles in row-major order (one-by-one tile row)
					 * compute dimension and allocate arrays asynchronously od devices */
					for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {

						/* Get the number of rows in the current tile */
						tile_x = get_tile_size_row(dev_x);

						for (int dev_y = 0; dev_y < ntile_n_; dev_y++) {

							/* Get the number of columns in the current tile */
							tile_y = get_tile_size_col(dev_y);

							/* Get the GPU id */
							int gpu_id = dev_x * ntile_n_ + dev_y;

							/* Set device */
							cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + gpu_id));

							//* Allocate memories for H, IMT and B matrices */
							cuda_exec(cudaMallocPitch((void**)&B_[gpu_id], &pitchB[gpu_id],  maxBlock_*sizeof(T), ldB));
							cuda_exec(cudaMallocPitch((void**)&IMT_[gpu_id], &pitchIMT[gpu_id], maxBlock_ * sizeof(T), ldIMT));
							cuda_exec(cudaMallocPitch((void**)&WRKSPACE_[gpu_id], &pitchWRK[gpu_id], maxBlock_ * sizeof(T), ldWRK));
							cuda_exec(cudaMallocPitch((void**)&H_[gpu_id], &pitchH[gpu_id], tile_y * sizeof(T), ldH));
						}
					}

	
					//for shifting matrix on gpus
					std::size_t start_row, start_col;
 					d_off_m_ = (std::size_t**) malloc(num_devices_per_rank * sizeof(std::size_t*));
					d_off_n_ = (std::size_t**) malloc(num_devices_per_rank * sizeof(std::size_t*));

					for (int dev_x = 0; dev_x < ntile_m_; dev_x++){
						tile_x = get_tile_size_row(dev_x);
						start_row = dev_x * dim_tile_m_;

						for(int dev_y = 0; dev_y < ntile_n_; dev_y++) {
							tile_y = get_tile_size_col(dev_y);
							start_col = dev_y * dim_tile_n_;
							int dev_id = dev_x * ntile_n_ + dev_y;
							std::vector<std::size_t> off_m, off_n;
                                                       
							for(std::size_t j = 0; j < nblocks_; j++){
								for(std::size_t i = 0; i < mblocks_; i++){
									for(std::size_t q = 0; q < c_lens_[j]; q++){
										for(std::size_t p = 0; p < r_lens_[i]; p++){

											if(q + c_offs_l_[j] >= start_col && q + c_offs_l_[j] < start_col + tile_y && p + r_offs_l_[i] >= start_row && p + r_offs_l_[i] < start_row + tile_x){
												std::size_t s, t;
												//t, s, global index
												t = q + c_offs_[j];
												s = p + r_offs_[i];

												if(t == s){
													off_m.push_back(p + r_offs_l_[i] - start_row);
													off_n.push_back(q + c_offs_l_[j] - start_col);
												}
											}

										}
									}
								}
							}

							std::size_t off_size = off_m.size();
							diagonal_offs_.push_back(off_size);
							cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + dev_id));
							cudaMalloc(&d_off_m_[dev_id], off_size * sizeof(std::size_t));
							cudaMalloc(&d_off_n_[dev_id], off_size * sizeof(std::size_t));
							cudaMemcpy(d_off_m_[dev_id], off_m.data(), off_size* sizeof(std::size_t), cudaMemcpyHostToDevice);
							cudaMemcpy(d_off_n_[dev_id], off_n.data(), off_size* sizeof(std::size_t), cudaMemcpyHostToDevice);

						}
					}


					//
					/* Keep info wether the matrix H_ is distributed among devices or not. In the initialization phase, it is not distributed */
					copied_ = false;
					next_ = NextOp::cAb;

					/* Set memory pointers. The initial configuration is: IMT = H * B + IMT, i.e. V = B and W = IMT */
					this->switch_pointers();

					/* Start timing events */
					//cuda_exec(cudaEventCreate(&start));
					//cuda_exec(cudaEventCreate(&stop));

					//Allocate device memory for other non-HEMM operations
					cudaSetDevice(shmrank_*num_devices_per_rank);
					cuda_exec(cudaMalloc ((void**)&d_V1_  , sizeof(T)*m_*(nev_ + nex_)));
                                        cuda_exec(cudaMalloc ((void**)&d_V2_  , sizeof(T)*m_*(nev_ + nex_)));
                                        cuda_exec(cudaMalloc ((void**)&d_A_  , sizeof(T)*(nev_ + nex_)*(nev_ + nex_)));					
					cuda_exec(cudaMalloc ((void**)&d_ritz_, sizeof(Base<T>) * (nev_ + nex_)));

					// Allocating workspace for xPOTRF and HEEVD
					cudaSetDevice(shmrank_*num_devices_per_rank);
					cuda_exec(cudaMalloc ((void**)&devInfo_, sizeof(int)));
					cudaSetDevice(shmrank_*num_devices_per_rank);
					cusolver_status_ = cusolverDnTpotrf_bufferSize(cusolverH_[0], 
							CUBLAS_FILL_MODE_UPPER, nev_ + nex_, d_A_, nev_ + nex_, &lwork_potrf);
					assert (cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
					cudaSetDevice(shmrank_*num_devices_per_rank);
					cusolver_status_ = cusolverDnTheevd_bufferSize(cusolverH_[0],CUSOLVER_EIG_MODE_VECTOR,
						CUBLAS_FILL_MODE_LOWER, nev_ + nex_, d_A_, nev_ + nex_, d_ritz_, &lwork_heevd);
					lwork_ = (lwork_potrf > lwork_heevd)? lwork_potrf : lwork_heevd;
                                        cudaSetDevice(shmrank_*num_devices_per_rank);					
					cuda_exec(cudaMalloc((void**)&d_work_, sizeof(T)*lwork_));

					/* Set time collectors to zero */
					time_copy_H = std::chrono::milliseconds::zero(); 
					time_copy_W = std::chrono::milliseconds::zero();
					time_copy_V = std::chrono::milliseconds::zero();
					time_gemm = std::chrono::milliseconds::zero();
					time_dist = std::chrono::milliseconds::zero();
					time_axpy = std::chrono::milliseconds::zero();
					time_redist = std::chrono::milliseconds::zero();
				}

				/* Remove local variables and free arrays */
				~mgpu_cudaDLA() { 

					for (int dev=0; dev<num_devices_per_rank; dev++) {
						cudaStreamDestroy(stream_[dev]);
                                                cudaStreamDestroy(stream2_[dev]);
						cublasDestroy(handle_[dev]);
						cudaFree(H_[dev]);
						cudaFree(B_[dev]);
						cudaFree(IMT_[dev]);
						cudaFree(WRKSPACE_[dev]);
				   	        if (cusolverH_[dev]) cusolverDnDestroy(cusolverH_[dev]);

						cudaFree(d_off_m_[dev]);
						cudaFree(d_off_n_[dev]);
					}
					free(B_);
					free(IMT_);
					free(WRKSPACE_);
					free(H_);
					free(V);
					free(W);
					free(pitchB);
					free(pitchH);
					free(pitchIMT);
					free(pitchWRK);
					free(pitchV);
					free(pitchW);

					if (devInfo_) cudaFree(devInfo_);
        				if (d_V1_) cudaFree(d_V1_);
                                        if (d_V2_) cudaFree(d_V2_);
                                        if (d_A_) cudaFree(d_A_);					
					//if (d_return_) cudaFree(d_return_);
        				if (d_work_) cudaFree(d_work_);	
					if (d_ritz_) cudaFree(d_ritz_);	
				}

				/* Distribute given matrix orig_H among GPUs */
				//! This member functions Distribute given matrix `orig_H` among GPUs
				/*!
					@param orig_H: original matrix to be distributed on Host
					@param ld_origH: leading dimension of `orig_H`.
				*/

				void distribute_H(T* orig_H, std::size_t ld_origH ) {

					int tile_x, tile_y;

					int start_row, start_col;

					/* If H is not distributed among devices (i.e. the first call to the distribute_H), distribute it */
					if( !copied_ ) {
						/* Pass through the rows of the tiled matrix H */
						for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {

							/* Get the number of rows in the tile */
							tile_x = get_tile_size_row(dev_x);

							/* Get the starting row-index of the tile */
							start_row = dev_x * dim_tile_m_;

							/* Pass through the columns of the tiled matrix H */
							for(int dev_y = 0; dev_y < ntile_n_; dev_y++) {

								/* FGet the number of columns in the tile */
								tile_y = get_tile_size_col(dev_y);

								/* Get the starting column-index of the tile */
								start_col = dev_y * dim_tile_n_;

								/* Compute the device id (ranging from 0..num_devices-1 */
								int dev_id = dev_x * ntile_n_ + dev_y;

								/* Set a device and asyncronously transfer the tile to that device's memory */
								cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + dev_id));

								cublas_exec(cublasSetMatrixAsync(tile_x, tile_y, sizeof(T), &orig_H[start_col * ld_origH + start_row], ld_origH, H_[dev_id], ldH, stream_[dev_id]));
							}
						}

						/* Next time the function is called, no need to distribute it again */
						copied_ = true;
					}
				}


				void shiftMatrix(T c, bool isunshift = false) {
					int tile_x, tile_y;
					int count_x = 0, count_y = 0;

					std::size_t start_row, start_col;

					for (int dev_x = 0; dev_x < ntile_m_; dev_x++){
						tile_x = get_tile_size_row(dev_x);
						start_row = dev_x * dim_tile_m_;

						for(int dev_y = 0; dev_y < ntile_n_; dev_y++) {
							tile_y = get_tile_size_col(dev_y);
							start_col = dev_y * dim_tile_n_;
							int dev_id = dev_x * ntile_n_ + dev_y;
							
							std::size_t off_size = diagonal_offs_[dev_id];

							cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + dev_id));
							chase_shift_mgpu_matrix(H_[dev_id], d_off_m_[dev_id], d_off_n_[dev_id], off_size, ldH, std::real(c), stream_[dev_id]);

						}
					}
				}

				/* Divide given matrix buf_init into panels and distributed among GPUs */
				//! This member function divides given matrix `buf_init` into panels and distributed among GPUs
				/*!
					@param buf_init: the given matrix to be divided into panels and distributed among GPUs
					@param ldBuf: the leading dimension of `buf_init`
					@param block: number of non-converged eigenvectors, it indicates the number of columns in `V` which performs the `HEMM` operation.
				*/
				void distribute_V (T* buf_init, std::size_t ldBuf, std::size_t block) {

					/* Number of rows in the tile */
					int tile_x;

					/* Index of the first row of the tile */
					int start_row;

					/* In the case of cAb operation, i.e. W = H * V + W */
					if (next_ == NextOp::cAb) {

						/* Divide buf_init in row-panels matching the number of tile-columns of tiled H */ 
						for (int i = 0; i < ntile_n_; i++) {

							/* Number of rows in the tile */
							tile_x = get_tile_size_col(i);

							/* Index of the first row in the tile in the global H indexing */
							start_row = i * dim_tile_n_;

							/* Pass through all devices */
							for (int dev = i; dev < num_devices_per_rank; dev += ntile_n_) {

								cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + dev));
								cublas_exec(cublasSetMatrixAsync(tile_x, block, sizeof(T), &buf_init[start_row], ldBuf, V[dev], ldV, stream_[dev]));
							}
						}
					}
					/* The case V = H^C * W + V */
					else {
						/* Divide buf_init input tile ntile_m_ tiles and distribute to GPU such as
 						 * that tile i goes to the GPUs with GPU_ID % ntile_m_ == i
 						 */ 
						/* Divide buf_init in row-panels matching the number of tile-rows of the tiled H (because H is (conjugate) transposed */ 
						for (int i = 0; i < ntile_m_; i++) {

							/* Number of rows in the tile */
							tile_x = get_tile_size_row(i);

							/* Index of the first row in the tile in the global H indexing */
							start_row = i * dim_tile_m_;
							
							/* Compute index of the first GPU device in the (transponsed) tile-row of the tiled matrix H */
							int start_dev_id = i * ntile_n_;

							/* Pass through all the devices in the tile-row */
							for (int dev = start_dev_id; dev < start_dev_id + ntile_n_; dev++) {

								cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + dev));
								cublas_exec(cublasSetMatrixAsync(tile_x, block, sizeof(T), &buf_init[start_row], ldBuf, V[dev], ldV, stream_[dev]));
							}
						}
					}
					
				}

				/* Compute Hemm */
				//! This member function computes HEMM within each GPU card.
				/*!
					@param block: number of non-converged eigenvectors, it indicates the number of columns in `V` which performs the `HEMM` operation.
					@param alpha&beta: scalars of type `T` in `HEMM` functions.
				*/
				void computeHemm(std::size_t block, std::size_t offset, T alpha, T beta) {

					/* Parameters for a local <T>hemm operation */
					std::size_t m, n, k;
					cublasOperation_t transa;

					/* Dimension of the tile */
					std::size_t tile_x, tile_y;
	
					/* Define '0' and '1' */
					T zero = T(0.0);
					T one = T(1.0);

					/* Auxiliary variables */
					bool leading_gpu = false;
					std::size_t num_tile_cols;

					/* Shift matrix W. W is holding B or C column-matrix from previous iteration */
					for( int gpu_id = 0; gpu_id < num_devices_per_rank; gpu_id++) {
						W[gpu_id] += offset * ldW;
					}

					/* Set transa (for H). In the case bAc H is (conjugate-)transposed */
					if (next_ == NextOp::bAc) {
						transa = CUBLAS_OP_C;
						num_tile_cols = ntile_m_;
					} else {
						transa = CUBLAS_OP_N;
						num_tile_cols = ntile_n_;
					}

					int dev_id; 

					/* Visit all the tiles of the tile matrix H and compute local (partial) HEMM operations */

					/* Pass through the rows of tiled H */
					for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {
						
						/* Number of rows in the tile */
						tile_x = get_tile_size_row(dev_x);

						/* Pass through the columns of tiled H */
						for (int dev_y = 0; dev_y < ntile_n_; dev_y++) {
						
							/* Number of columns in the tile */
							tile_y = get_tile_size_col(dev_y);

							/* Get m, n, k for each operation cAb and bAc. 
 							 * The first GPU of each row stored intermediat results in W, others in the WRKSPACE array */
							if (next_ == NextOp::cAb) {
								m = tile_x;
								n = block;
								k = tile_y;
								if(dev_y == 0) {
									leading_gpu = true;
								} else {
									leading_gpu = false;
								}
								dev_id = dev_x * ntile_n_ + dev_y;
							} else {
								m = tile_y;
								n = block;
								k = tile_x;
								if(dev_x == 0) {
									leading_gpu = true;
								} else {
									leading_gpu = false;
								}
								dev_id = dev_x * ntile_n_ + dev_y;
							}
		
							/* Set device and compute local Tgemm. The GPUs operating on the the first tiles in the rows are updating
 							 * W, while other GPUs store intermediate results in the cleaned (multiplied by '0') WRKSPACE array */
							cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + dev_id));
							if (leading_gpu) {
								cublas_exec(cublasTgemm(handle_[dev_id], transa, CUBLAS_OP_N, m, n, k, &alpha, H_[dev_id], ldH, V[dev_id], ldV,
									&beta, W[dev_id], ldW));
							} else {
								cublas_exec(cublasTgemm(handle_[dev_id], transa, CUBLAS_OP_N, m, n, k, &alpha, H_[dev_id], ldH, V[dev_id], ldV,
									&zero, WRKSPACE_[dev_id], ldWRK));
							}
						}
					}
					
					/* Synchronize all GPUs */
					this->synchronizeAll();

					int gpu_src;
					int gpu_dest;

					/* Compute the final solution from partial per-GPU solutions.
					 * Collect intermediate results from the GPUs in the same rows to the first GPU in the row.
					 * Implemented as a parallel prefix sum. 
					 * In the first step the odd GPUs transfer their partial solutions (tile products) to a one previous GPU (i.e. the one with the index-1) where the
					 * sum of two tiles are computed and so on. */ 
					for (int s = 1; s < num_tile_cols; s <<= 1) {

						if (next_ == NextOp::cAb) {
							for (int dev = s; dev < num_devices_per_rank; dev += 2*s) {
								tile_x = get_tile_size_row(dev/ntile_n_);

								if (s == 1 || (num_tile_cols%2 != 0 && dev == num_devices_per_rank-1)) {
									cuda_exec(cudaMemcpyAsync(WRKSPACE_[dev-s], WRKSPACE_[dev], block*sizeof(T)*ldWRK, cudaMemcpyDeviceToDevice, stream_[dev-s]));
								} else {
									cuda_exec(cudaMemcpyAsync(WRKSPACE_[dev-s], W[dev], block*sizeof(T)*ldW, cudaMemcpyDeviceToDevice, stream_[dev-s]));
								}
								cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + dev-s));

								cublas_exec(cublasTaxpy(handle_[dev-s], block*ldW, &one, WRKSPACE_[dev-s], 1, W[dev-s], 1));
							}
						} else {
							for (int dev_x = 0; dev_x < ntile_n_; dev_x++) {
								for (int dev_y = dev_x+s*ntile_n_; dev_y < num_devices_per_rank; dev_y += 2*s*ntile_n_) {
									gpu_src = dev_y;
									gpu_dest = dev_y - s*ntile_n_;

									tile_x = get_tile_size_row(dev_y/ntile_m_);
									if (s == 1 || (num_tile_cols%2 != 0 && dev_y/ntile_n_ == ntile_m_-1)) {
										cuda_exec(cudaMemcpyAsync(WRKSPACE_[gpu_dest], WRKSPACE_[gpu_src], block*sizeof(T)*ldWRK, cudaMemcpyDeviceToDevice, stream_[gpu_dest]));
									} else {
										cuda_exec(cudaMemcpyAsync(WRKSPACE_[gpu_dest], W[gpu_src], block*sizeof(T)*ldW, cudaMemcpyDeviceToDevice, stream_[gpu_dest]));
									}
									cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + gpu_dest));

									cublas_exec(cublasTaxpy(handle_[gpu_dest], block*ldW, &one, WRKSPACE_[gpu_dest], 1, W[gpu_dest], 1));
								}
							}
						}
					}
				}

				/* Collect and return the computed W from the GPUs to the host*/
				//! This member function collects and returns the computed `W` from the GPUs to the host
				/*!
					@param buf_target: the matrix host which the computed `W` should be copied to.
					@param ldBuf: the leading dimension of `buf_target`
					@param block: number of non-converged eigenvectors, it indicates the number of columns in `V` which performs the `HEMM` operation.
				*/
				void return_W (T* buf_target, std::size_t ldBuf, std::size_t block, std::size_t offset) {

					/*  */
					int tile_x;
					int start_row;
					int src_gpu;

					if (next_ == NextOp::cAb) {
						for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {
							src_gpu = dev_x * ntile_n_;
							tile_x = get_tile_size_row(dev_x);
							start_row = dev_x * dim_tile_m_;

							cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + src_gpu));
							cublas_exec(cublasGetMatrixAsync(tile_x, block, sizeof(T), W[src_gpu], ldW, &buf_target[start_row], ldBuf, stream_[src_gpu]));
						}

					} else {
						for (int dev_x = 0; dev_x < ntile_n_; dev_x++) {
							src_gpu = dev_x;
							tile_x = get_tile_size_col(dev_x);
							start_row = dev_x * dim_tile_n_;

							cuda_exec(cudaSetDevice(shmrank_*num_devices_per_rank + src_gpu));
							cublas_exec(cublasGetMatrixAsync(tile_x, block, sizeof(T), W[src_gpu], ldW, &buf_target[start_row], ldBuf, stream_[src_gpu]));
						}

					}

				}

				/* Synchronize all devices */
				//! This member function synchronizes all devices
				void synchronizeAll() {

					for (int i = 0; i < num_devices_per_rank; i++) {
						cudaStreamSynchronize(stream_[i]);
					}
				}

				/* Switch pointers to per-device 2D arrays depending on the next operation (cAb or bAc) */
				//! This member function switches pointers to per-device 2D arrays depending on the next operation (cAb or bAc)
				void switch_pointers(){

					this->synchronizeAll();
					for(int gpu_id = 0; gpu_id < num_devices_per_rank; gpu_id++) {
						/* If next operation is set to W = H * V + W */
						if (next_ == NextOp::cAb) {
							V[gpu_id] = B_[gpu_id];
							W[gpu_id] = IMT_[gpu_id];
							
							pitchV[gpu_id] = pitchB[gpu_id];
							pitchW[gpu_id] = pitchIMT[gpu_id];

							ldV = ldB;
							ldW = ldIMT;
						/* Computation direction is V = H^C * W + V */
						} else {
							W[gpu_id] = B_[gpu_id];
							V[gpu_id] = IMT_[gpu_id];
							
							pitchW[gpu_id] = pitchB[gpu_id];
							pitchV[gpu_id] = pitchIMT[gpu_id];

							ldW = ldB;
							ldV = ldIMT;
						}	
					}
					
				}

				/* Switch operation, between W = H * V + W and V = H^C W + V */
				//! This function defines a switch operation, between W = H * V + W and V = H^C W + V 
				void switch_operation() {

					/* Change operation */
					if ( next_ == NextOp::bAc) {
						next_ = NextOp::cAb;
					} else {
						next_ = NextOp::bAc;
					}

					/* Switch pointers */
					this->switch_pointers();
				}

				/* Set operation, bAc or cAb */
				void set_operation(int next) {

					if(next == 0) {
						next_ = NextOp::cAb;
					} else if(next == 1) {
						next_ = NextOp::bAc;
					} else {
						std::cout << "Wrong operation identifier!" << std::endl;
					}
					this->switch_pointers();
				}

				void gemm_small(std::size_t m,
                         			std::size_t n, std::size_t k, T* alpha, T* a, std::size_t lda, T* b,
                         				std::size_t ldb, T* beta, T* c, std::size_t ldc){
				   

				    cudaDeviceSynchronize();
			    	    cudaSetDevice(shmrank_*num_devices_per_rank);
				    cublas_status_ = cublasSetMatrixAsync(k, m, sizeof(T), a, lda, d_V1_, m_, stream_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                    cublas_status_ = cublasSetMatrixAsync(k, n, sizeof(T), b, ldb, d_V2_, m_, stream_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                    cublas_status_ = cublasSetMatrixAsync(m, n, sizeof(T), c, ldc, d_A_, (nev_+nex_), stream_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
				    cudaSetDevice(shmrank_*num_devices_per_rank);

				    if(sizeof(T) == sizeof(Base<T>)){

				    	cublas_status_ = cublasTgemm(handle_[0], CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, d_V1_, m_, 
						    	d_V2_, m_, beta, d_A_, nev_+nex_); 
				    }
				    else{

                                        cublas_status_ = cublasTgemm(handle_[0], CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, alpha, d_V1_, m_,
                                                        d_V2_, m_, beta, d_A_, nev_+nex_);

				    }
				   
				    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

                                    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cublas_status_ = cublasGetMatrixAsync(m, n, sizeof(T), d_A_, (nev_+nex_), c, ldc, stream_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
				    cudaDeviceSynchronize();

				}

                                void gemm_large(std::size_t m,
                                                std::size_t n, std::size_t k, T* alpha, T* a, std::size_t lda, T* b,
                                                        std::size_t ldb, T* beta, T* c, std::size_t ldc){
                                    cudaDeviceSynchronize();
                                    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cublas_status_ = cublasSetMatrixAsync(m, k, sizeof(T), a, lda, d_V1_, m_, stream_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                    cublas_status_ = cublasSetMatrixAsync(k, n, sizeof(T), b, ldb, d_A_, (nev_+nex_), stream_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                    cublas_status_ = cublasSetMatrixAsync(m, n, sizeof(T), c, ldc, d_V2_, m_, stream_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

				    cublas_status_ = cublasTgemm(handle_[0],CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_V1_, m_, 
						    			d_A_, nev_+nex_, beta, d_V2_, m_);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
				    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cublas_status_ = cublasGetMatrixAsync(m, n, sizeof(T), d_V2_, m_, c, ldc, stream_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
				}

				int potrf(char uplo, std::size_t n, T* a, std::size_t lda){
				    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cublas_status_ = cublasSetMatrixAsync(n, n, sizeof(T), a, lda, d_A_, (nev_+nex_), stream2_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                    cudaSetDevice(shmrank_*num_devices_per_rank);
    				    cusolverDnTpotrf(cusolverH_[0], CUBLAS_FILL_MODE_UPPER, nev_+nex_, d_A_, 
						    			nev_+nex_, d_work_, lwork_, devInfo_);				    
				    int info;
                                    cudaSetDevice(shmrank_*num_devices_per_rank);				    
				    cuda_exec(cudaMemcpy(&info, devInfo_, 1 * sizeof(int), cudaMemcpyDeviceToHost));
                                    cublas_status_ = cublasGetMatrixAsync(n, n, sizeof(T), d_A_, (nev_+nex_), a, lda, stream2_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);				 
				    return info;
				}

				//later this heevd should be implemeted togather with the second gemm for backtransformation
				void heevd(std::size_t n, T* a, std::size_t lda, Base<T>* w){
				    cudaSetDevice(shmrank_*num_devices_per_rank);
				    cublas_status_ = cublasSetMatrixAsync(n, n, sizeof(T), a, lda, d_A_, (nev_+nex_), stream2_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                    cudaSetDevice(shmrank_*num_devices_per_rank);
				    cusolver_status_ = cusolverDnTheevd(cusolverH_[0],CUSOLVER_EIG_MODE_VECTOR,
						  CUBLAS_FILL_MODE_LOWER, n, d_A_, (nev_+nex_), d_ritz_, d_work_, lwork_,devInfo_);

				    assert (cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
				    cudaSetDevice(shmrank_*num_devices_per_rank);
				    cuda_exec(cudaMemcpyAsync(w, d_ritz_, sizeof(Base<T>) * n, cudaMemcpyDeviceToHost, stream2_[0]));
                                    cublas_status_ = cublasGetMatrixAsync(n, n, sizeof(T), d_A_, (nev_+nex_), a, lda, stream2_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);				    
				}

 				void heevd2(std::size_t m_, std::size_t block, T* A, std::size_t lda, T *approxV, std::size_t ldv, 
						T* workspace, std::size_t ldw, std::size_t offset, Base<T>* ritzv)
				{

				    T One = T(1.0);
				    T Zero = T(0.0);

				    //heevd
				    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cublas_status_ = cublasSetMatrixAsync(block, block, sizeof(T), A, lda, d_A_, nev_+nex_, stream2_[0]);
				    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
				    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cusolver_status_ = cusolverDnTheevd(cusolverH_[0],CUSOLVER_EIG_MODE_VECTOR,
                                                  CUBLAS_FILL_MODE_LOWER, block, d_A_, (nev_+nex_), d_ritz_, d_work_, lwork_heevd,devInfo_);

				    assert (cusolver_status_ == CUSOLVER_STATUS_SUCCESS);
				    cudaSetDevice(shmrank_*num_devices_per_rank);
				    cuda_exec(cudaMemcpyAsync(ritzv, d_ritz_, sizeof(Base<T>) * block, cudaMemcpyDeviceToHost, stream2_[0]));

				    //gemm
				    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cublas_status_ = cublasSetMatrixAsync(m_, block, sizeof(T), approxV + offset, ldv, d_V1_, m_, stream_[0]);				    
				    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
				    cudaSetDevice(shmrank_*num_devices_per_rank);
				    cublas_status_ = cublasTgemm(handle_[0],CUBLAS_OP_N, CUBLAS_OP_N, m_, block, block, &One, 
						    	d_V1_, m_, d_A_, nev_+nex_, &Zero, d_V2_, m_);
				    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cublas_status_ = cublasGetMatrixAsync(m_, block, sizeof(T), d_V2_, m_, workspace + offset, ldw, stream_[0]);				    

				}

				void syherk(std::size_t n, std::size_t k, T* a, std::size_t lda, T* c, std::size_t ldc){

				//t_syherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
 				    Base<T> One = 1.0;
				    Base<T> Zero = 0.0;

				    cudaSetDevice(shmrank_*num_devices_per_rank);
				    cublas_status_ = cublasSetMatrixAsync(k, n, sizeof(T), a, lda, d_V1_, m_, stream_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);				    
                                    cudaSetDevice(shmrank_*num_devices_per_rank);
				    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                    cublas_status_ = cublasSetMatrixAsync(n, n, sizeof(T), c, ldc, d_A_, nev_ + nex_, stream_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
				    if(sizeof(T) == sizeof(Base<T>))
				        cublas_status_ = cublasTsyherk(handle_[0], CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, k, &One, 
						    d_V1_, m_, &Zero, d_A_, nev_ + nex_);
				    else
                                        cublas_status_ = cublasTsyherk(handle_[0], CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, n, k, &One,
                                                    d_V1_, m_, &Zero, d_A_, nev_ + nex_);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
				    cublas_status_ = cublasGetMatrixAsync(n, n, sizeof(T), d_A_, nev_ + nex_, c, ldc, stream_[0]);
				}

  				int shiftedcholQR(std::size_t m_, std::size_t nevex, T *approxV, std::size_t ldv, 
						T *A, std::size_t lda, std::size_t offset) {
 			            int info = -1;
      				    int grank;
      				    MPI_Comm_rank(MPI_COMM_WORLD, &grank);

                                    T one = T(1.0);
                                    T zero = T(0.0);
                                    //potrf
                                    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cublas_status_ = cublasSetMatrixAsync(nevex, nevex, sizeof(T), A, lda, d_A_, nev_+nex_, stream2_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cusolverDnTpotrf(cusolverH_[0], CUBLAS_FILL_MODE_UPPER, nev_+nex_, d_A_,
                                                                        nev_+nex_, d_work_, lwork_potrf, devInfo_);

                                    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cuda_exec(cudaMemcpy(&info, devInfo_, 1 * sizeof(int), cudaMemcpyDeviceToHost));

				    if (info != 0){
				        //first CholeskyQR with shift: https://doi.org/10.1137/18M1218212
					Base<T> normV = t_lange('F', ldv, nevex, approxV, ldv);
					// generate shift
					std::size_t mul = ldv * nevex + nevex * nevex + nevex;
					Base<T> s = 11.0 * static_cast<Base<T>>(mul) * std::numeric_limits<Base<T>>::epsilon() * normV;
#if defined(CHASE_OUTPUT)
					if(grank == 0){
					    std::cout << "Cholesky Factorization is failed for QR, a shift is performed: " << s << std::endl;
					}
#endif					
                                        //shift matrix A with s on host
			       		for(std::size_t i = 0; i < nevex; i++){
             				    A[i * nevex + i] = A[i * nevex + i] + s;
					}
					//re-offload new shifted A to GPU
                                    	cudaSetDevice(shmrank_*num_devices_per_rank);
                                        cublas_status_ = cublasSetMatrixAsync(nevex, nevex, sizeof(T), A, lda, d_A_, nev_+nex_, stream2_[0]);
                                        assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                        //perform Cholesky factorization with shifted A 
					cudaSetDevice(shmrank_*num_devices_per_rank);
                                        cusolverDnTpotrf(cusolverH_[0], CUBLAS_FILL_MODE_UPPER, nev_+nex_, d_A_,
                                                                        nev_+nex_, d_work_, lwork_potrf, devInfo_);					
				    }
                                    //trsm
                                    cudaSetDevice(shmrank_*num_devices_per_rank);
                                    cublas_status_ = cublasSetMatrixAsync(m_, nevex, sizeof(T), approxV + offset, ldv, d_V1_, m_, stream2_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                    cublas_status_ = cublasTtrsm(handle_[0], CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                                                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m_, nevex, &one, d_A_,
                                                                        nev_+nex_, d_V1_, m_);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
                                    cublas_status_ = cublasGetMatrixAsync(m_, nevex, sizeof(T), d_V1_, m_, approxV + offset, ldv, stream2_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

				    return info;
			       	}	


                                int cholQR(std::size_t m_, std::size_t nevex, T *approxV, std::size_t ldv,
                                                T *A, std::size_t lda, std::size_t offset)  {
                                    int info = -1;

				    T one = T(1.0);
				    T zero = T(0.0);
			            //potrf
				    cudaSetDevice(shmrank_*num_devices_per_rank);
				    cublas_status_ = cublasSetMatrixAsync(nevex, nevex, sizeof(T), A, lda, d_A_, nev_+nex_, stream2_[0]);
				    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
				    cudaSetDevice(shmrank_*num_devices_per_rank);
    				    cusolverDnTpotrf(cusolverH_[0], CUBLAS_FILL_MODE_UPPER, nev_+nex_, d_A_, 
						    			nev_+nex_, d_work_, lwork_potrf, devInfo_);

				    cudaSetDevice(shmrank_*num_devices_per_rank);
				    cuda_exec(cudaMemcpy(&info, devInfo_, 1 * sizeof(int), cudaMemcpyDeviceToHost));
				    
				    //trsm
				    cudaSetDevice(shmrank_*num_devices_per_rank);
				    cublas_status_ = cublasSetMatrixAsync(m_, nevex, sizeof(T), approxV + offset, ldv, d_V1_, m_, stream2_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);
				    cublas_status_ = cublasTtrsm(handle_[0], CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, 
						    			CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m_, nevex, &one, d_A_, 
									nev_+nex_, d_V1_, m_);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);				    
                                    cublas_status_ = cublasGetMatrixAsync(m_, nevex, sizeof(T), d_V1_, m_, approxV + offset, ldv, stream2_[0]);
                                    assert(cublas_status_ == CUBLAS_STATUS_SUCCESS);

                                    return info;
                                }

			private:

				/// Dimension of the input matrices H (m*n), W(m*maxBlock) and V(n*maxBlock)
				std::size_t n_;
				std::size_t m_;
				std::size_t maxBlock_;
				std::size_t max_dim_;
			  	std::size_t nev_;
  				std::size_t nex_;
                                std::size_t N_;

				/// Storage spaces
				T** B_ = nullptr;
				T** IMT_ = nullptr;
				T** H_ = nullptr;
				T** WRKSPACE_ = nullptr;

				/// Hemm pointers to operands W = A V + W
				T** W = nullptr;
				T** V = nullptr;

				/// Leading dimensions
				std::size_t ldB;
				std::size_t ldIMT;
				std::size_t ldWRK;
				std::size_t ldH;

				std::size_t ldV;
				std::size_t ldW;
				
				/// Pitched values
				std::size_t *pitchB = nullptr;
				std::size_t *pitchIMT = nullptr;
				std::size_t *pitchWRK = nullptr;
				std::size_t *pitchH = nullptr;
				
				std::size_t *pitchV = nullptr;
				std::size_t *pitchW = nullptr;

				bool copied_;

				enum NextOp { cAb, bAc };

				/// Keep the record of the next operation type (c = A b + c or c = b A^T +c )
				NextOp next_;

				/// List of GPU devices
				int num_devices;
				int num_devices_per_rank;

				/// Number of tiles in m/n direction
				int ntile_m_;
				int ntile_n_;

				/// Size of the tiles
				int dim_tile_m_;
				int dim_tile_n_;

  				/// Timing variables
				std::chrono::duration<double, std::milli> time_copy_H;
				std::chrono::duration<double, std::milli> time_copy_W;
				std::chrono::duration<double, std::milli> time_copy_V;
				std::chrono::duration<double, std::milli> time_gemm, time_dist, time_axpy, time_redist;
				std::chrono::duration<double, std::milli> local_time;

				/// MPI ranks and communication
				MPI_Comm shmcomm;
				int shmsize_;
				int shmrank_;
				int globalrank_;

				/// Cublas handler
				cublasHandle_t *handle_ = nullptr;
			
				/// Cuda streams
				cudaStream_t *stream_ = nullptr;

				/// Cublas error
				cublasStatus_t cublasError;

				cublasStatus_t cublas_status_ = CUBLAS_STATUS_SUCCESS;
        			cusolverStatus_t cusolver_status_ = CUSOLVER_STATUS_SUCCESS;
				cusolverDnHandle_t *cusolverH_ = nullptr;
                                cudaStream_t *stream2_ = nullptr;

				int *devInfo_ = NULL;
      				T *d_V1_ = NULL;
                                T *d_V2_ = NULL;
                                T *d_A_ = NULL;				
				//T *d_return_ = NULL;
				T *d_work_ = NULL;
        			int lwork_ = 0;
				Base<T> *d_ritz_ = NULL;
				int lwork_potrf, lwork_heevd;

			  	std::size_t *r_offs_;
  			        std::size_t *r_lens_;
  				std::size_t *r_offs_l_;
  				std::size_t *c_offs_;
  				std::size_t *c_lens_;
  				std::size_t *c_offs_l_;
  				std::size_t nb_;
  				std::size_t mb_;
  				std::size_t nblocks_;
  				std::size_t mblocks_;

				//for shifting matrix on gpus
				std::size_t **d_off_m_ = nullptr;
				std::size_t  **d_off_n_ = nullptr;
				std::vector<std::size_t> diagonal_offs_;
				
				/// Return the number of rows of the tile with row-index 'tile_position'
				int get_tile_size_row (int tile_position) {
					if (tile_position + 1 == ntile_m_) {
						return m_ - tile_position * dim_tile_m_;
					} else {
						return dim_tile_m_;
					}
				}

				/// Return the number of columns of the tile with column-index 'tile_position'
				int get_tile_size_col (int tile_position) {
					if (tile_position + 1 == ntile_n_) {
						return n_ - tile_position * dim_tile_n_;
					} else {
						return dim_tile_n_;
					}
				}
		};
	}  // namespace matrixfree
}  // namespace chase


