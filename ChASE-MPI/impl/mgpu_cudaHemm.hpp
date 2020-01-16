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
#include <iostream>

#include <chrono>

#include "algorithm/types.hpp"
#include "chase_mpi_properties.hpp"
#include "blas_cuda_wrapper.hpp"

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

		template <class T>
		class mgpu_cudaHemm {

			public:

				typedef T value_type;

				/* Constructor - sets GPUs, allocate device arrays, create cublas handles and streams */
				mgpu_cudaHemm() {};

				mgpu_cudaHemm(std::size_t m, std::size_t n, std::size_t maxBlock) :
				m_(m), n_(n), maxBlock_(maxBlock) {
				
					/* Get number of available GPU devices */
					cuda_exec(cudaGetDeviceCount(&num_devices));

					/* Allocate array for device indices, handles and streams */
					handle_ = (cublasHandle_t*) malloc(num_devices * sizeof(cublasHandle_t));
					stream_ = (cudaStream_t*) malloc(num_devices * sizeof(cudaStream_t));

					/* Populate list of devices, create handles and streams for each device */
					for (int dev=0; dev<num_devices; dev++) {
						cuda_exec(cudaSetDevice(dev));
						cublasCreate(&handle_[dev]);
						cuda_exec(cudaStreamCreate(&stream_[dev]));
						cublasSetStream(handle_[dev], stream_[dev]);
					}

					/* Allocate arrays to hold pointers to memory allocations on each device for
					 * matrices H, B and IMT */
					H_ = (T**) malloc(num_devices * sizeof(T*));
					B_ = (T**) malloc(num_devices * sizeof(T*));
					IMT_ = (T**) malloc(num_devices * sizeof(T*));
					WRKSPACE_ = (T**) malloc(num_devices * sizeof(T*));

					/* Pointers to memory location of the HEMM operands. The class always computes W = H * V + W operation */
					W = (T**) malloc(num_devices * sizeof(T*));
					V = (T**) malloc(num_devices * sizeof(T*));
					
					/* Allocate arrays to hold pitch values for each 2D array */
					pitchB = (std::size_t*) malloc(num_devices * sizeof(std::size_t));
					pitchH = (std::size_t*) malloc(num_devices * sizeof(std::size_t));
					pitchIMT = (std::size_t*) malloc(num_devices * sizeof(std::size_t));
					pitchWRK = (std::size_t*) malloc(num_devices * sizeof(std::size_t));

					/* Pointers to pitch values of the HEMM operands */
					pitchW = (std::size_t*) malloc(num_devices * sizeof(std::size_t));
					pitchV = (std::size_t*) malloc(num_devices * sizeof(std::size_t));

					/* Compute number of tiles of matrix H
					 * The number of tiles depends on number of the available GPU devices */
					ntile_n_ = sqrt(num_devices);
					ntile_m_ = num_devices/ntile_n_;

					/* Compute tile dimensions */
					dim_tile_n_ = std::min(n_, (n_+ntile_n_-1)/ntile_n_);
					dim_tile_m_ = std::min(m_, (m_+ntile_m_-1)/ntile_m_);
					
					/* Set leading dimensions of the GPU arrays */
					/// TODO: ldB and ldIMT does not to be large as ldWRK but rather dim_tile_m and dim_tile_n_, respectively
					ldWRK = std::max(dim_tile_m_, dim_tile_n_);
					ldB = ldIMT = ldWRK;
					ldH = dim_tile_m_;

					std::cout << "Number of tiles: "<<ntile_m_ << " x " << ntile_n_ << std::endl;
					std::cout << "Tile dimension: "<<dim_tile_m_ << " x " << dim_tile_n_ << std::endl;

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
							cuda_exec(cudaSetDevice(gpu_id));

							//* Allocate memories for H, IMT and B matrices */
							cuda_exec(cudaMallocPitch((void**)&B_[gpu_id], &pitchB[gpu_id],  maxBlock_*sizeof(T), ldB));
							cuda_exec(cudaMallocPitch((void**)&IMT_[gpu_id], &pitchIMT[gpu_id], maxBlock_ * sizeof(T), ldIMT));
							cuda_exec(cudaMallocPitch((void**)&WRKSPACE_[gpu_id], &pitchWRK[gpu_id], maxBlock_ * sizeof(T), ldWRK));
							cuda_exec(cudaMallocPitch((void**)&H_[gpu_id], &pitchH[gpu_id], tile_y * sizeof(T), ldH));
						}
					}

					/* Keep info wether the matrix H_ is distributed among devices or not. In the initialization phase, it is not distributed */
					copied_ = false;
					next_ = NextOp::cAb;

					/* Set memory pointers. The initial configuration is: IMT = H * B + IMT, i.e. V = B and W = IMT */
					this->switch_pointers();

					/* Start timing events */
					//cuda_exec(cudaEventCreate(&start));
					//cuda_exec(cudaEventCreate(&stop));

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
				~mgpu_cudaHemm() { 

					for (int dev=0; dev<num_devices; dev++) {
						cudaStreamDestroy(stream_[dev]);
						cublasDestroy(handle_[dev]);
						cudaFree(H_[dev]);
						cudaFree(B_[dev]);
						cudaFree(IMT_[dev]);
						cudaFree(WRKSPACE_[dev]);
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

					//std::cout << "HEMM timings: " << std::endl;
					//std::cout << "Copy H   = " << time_copy_H.count()/1000 << " sec" << std::endl;
					//std::cout << "Copy V   = " << time_copy_V.count()/1000 << " sec" << std::endl;
					//std::cout << "Return W = " << time_copy_W.count()/1000 << " sec"   << std::endl;
					//std::cout << "Hemm     = " << (time_gemm.count()+time_dist.count()+time_axpy.count()+time_redist.count())/1000 << " sec"  << std::endl;
					//std::cout << "\t gemm   = " << time_gemm.count()/1000 << " sec"  << std::endl;
					//std::cout << "\t dist   = " << time_dist.count()/1000 << " sec"  << std::endl;
					//std::cout << "\t axpy   = " << time_axpy.count()/1000 << " sec"   << std::endl;
					//std::cout << "\t redist = " << time_redist.count()/1000 << " sec"   << std::endl;
					//std::cout << std::endl;

					//cudaEventDestroy(start);
					//cudaEventDestroy(stop);
				}

				/* Distribute given matrix orig_H among GPUs */
				void distribute_H(T* orig_H, std::size_t ld_origH) {

					/* Tile dimension */
					int tile_x, tile_y;

					/* Start row/col position in the given matrix */
					int start_row, start_col;

					//auto start = high_resolution_clock::now();

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
								cuda_exec(cudaSetDevice(dev_id));
								cublasError = cublasSetMatrixAsync(tile_x, tile_y, sizeof(T), &orig_H[start_col * ld_origH + start_row], ld_origH, H_[dev_id], ldH, stream_[dev_id]);
								if(cublasError != CUBLAS_STATUS_SUCCESS) {
									std::cout << "Error in cublasSetMatrixAsync H" << std::endl;
								}
							}
						}
						/* Next time the function is called, no need to distribute it again */
						copied_ = true;
					}
	
					/* If H already distributed, then don't have to do nothing. */
					/// TODO: Currently, the shift of H is done on CPU and there is no "smart" way to update it on devices. Therefore, for now, the H is redistributed
					else {
						for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {
							tile_x = get_tile_size_row(dev_x);
							start_row = dev_x * dim_tile_m_;

							for(int dev_y = 0; dev_y < ntile_n_; dev_y++) {
								tile_y = get_tile_size_col(dev_y);
								start_col = dev_y * dim_tile_n_;

								int dev_id = dev_x * ntile_n_ + dev_y;
								cuda_exec(cudaSetDevice(dev_id));
								cublasError = cublasSetMatrixAsync(tile_x, tile_y, sizeof(T), &orig_H[start_col * ld_origH + start_row], ld_origH, H_[dev_id], ldH, stream_[dev_id]);
								if(cublasError != CUBLAS_STATUS_SUCCESS) {
									std::cout << "Error in cublasSetMatrixAsync H" << std::endl;
								}
							}
						}

					}

					//this->synchronizeAll();
					//auto stop = high_resolution_clock::now();
					//time_copy_H += stop -start;
				}

				/* Divide given matrix buf_init into panels and distributed among GPUs */
				void distribute_V (T* buf_init, std::size_t ldBuf, std::size_t block) {

					/* Number of rows in the tile */
					int tile_x;

					/* Index of the first row of the tile */
					int start_row;

					//auto start = high_resolution_clock::now();

					/* In the case of cAb operation, i.e. W = H * V + W */
					if (next_ == NextOp::cAb) {

						/* Divide buf_init in row-panels matching the number of tile-columns of tiled H */ 
						for (int i = 0; i < ntile_n_; i++) {

							/* Number of rows in the tile */
							tile_x = get_tile_size_col(i);

							/* Index of the first row in the tile in the global H indexing */
							start_row = i * dim_tile_n_;

							/* Pass through all devices */
							for (int dev = i; dev < num_devices; dev += ntile_n_) {

								cuda_exec(cudaSetDevice(dev));
								cublasError = cublasSetMatrixAsync(tile_x, block, sizeof(T), &buf_init[start_row], ldBuf, V[dev], ldV, stream_[dev]);
								if(cublasError != CUBLAS_STATUS_SUCCESS) {
									std::cout << "Error in cublasSetMatrixAsync V/W" << " in cAb" << std::endl;
								}

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

								cuda_exec(cudaSetDevice(dev));
								cublasError = cublasSetMatrixAsync(tile_x, block, sizeof(T), &buf_init[start_row], ldBuf, V[dev], ldV, stream_[dev]);
								if(cublasError != CUBLAS_STATUS_SUCCESS) {
									std::cout << "Error in cublasSetMatrixAsync V/W" << " in bAc" << std::endl;
								}
							}
						}
					}

					//this->synchronizeAll();
					//auto stop = high_resolution_clock::now();
					//time_copy_V += stop - start;
					
				}

				/* Compute Hemm */
				void computeHemm(std::size_t block, T alpha, T beta) {

					/* Parameters for a local <T>hemm operation */
					std::size_t m, n, k;
					cublasOperation_t transa;

					/* Dimension of the tile */
					std::size_t tile_x, tile_y;
	
					/* Define '0' and '1' */
					T zero = 0;
					T one = 1;

					/* Auxiliary variables */
					bool leading_gpu = false;
					std::size_t num_tile_cols;

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

					//auto start = high_resolution_clock::now();

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
							cuda_exec(cudaSetDevice(dev_id));
							if (leading_gpu) {
								cublasTgemm(handle_[dev_id], transa, CUBLAS_OP_N, m, n, k, &alpha, H_[dev_id], ldH, V[dev_id], ldV,
									&beta, W[dev_id], ldW);
							} else {
								cublasTgemm(handle_[dev_id], transa, CUBLAS_OP_N, m, n, k, &alpha, H_[dev_id], ldH, V[dev_id], ldV,
									&zero, WRKSPACE_[dev_id], ldWRK);
							}
						}
					}
					
					/* Synchronize all GPUs */
					this->synchronizeAll();

					//auto stop = high_resolution_clock::now();
					//time_gemm += stop - start;

					int gpu_src;
					int gpu_dest;

					//start = high_resolution_clock::now();

					/* Compute the final solution from partial per-GPU solutions.
					 * Collect intermediate results from the GPUs in the same rows to the first GPU in the row.
					 * Implemented as a parallel prefix sum. 
					 * In the first step the odd GPUs transfer their partial solutions (tile products) to a one previos GPU (i.e. the one with the index-1) where the
					 * sum of two tiles are computed and so on. */ 
					for (int s = 1; s < num_tile_cols; s <<= 1) {

						if (next_ == NextOp::cAb) {
							for (int dev = s; dev < num_devices; dev += 2*s) {
								tile_x = get_tile_size_row(dev/ntile_n_);

								if (s == 1 || (num_tile_cols%2 != 0 && dev == num_devices-1)) {
									cuda_exec(cudaMemcpyAsync(WRKSPACE_[dev-s], WRKSPACE_[dev], block*sizeof(T)*ldWRK, cudaMemcpyDeviceToDevice, stream_[dev-s]));
								} else {
									cuda_exec(cudaMemcpyAsync(WRKSPACE_[dev-s], W[dev], block*sizeof(T)*ldW, cudaMemcpyDeviceToDevice, stream_[dev-s]));
								}
								cuda_exec(cudaSetDevice(dev-s));	
								cuda_exec(cudaStreamSynchronize(stream_[dev-s]));
								cublasError = cublasTaxpy(handle_[dev-s], (pitchWRK[dev-s]/sizeof(T))*ldWRK, &one, WRKSPACE_[dev-s], 1, W[dev-s], 1);
								if(cublasError != CUBLAS_STATUS_SUCCESS) std::cout << "Error in Taxpy " << std::endl;
							}
						} else {
							for (int dev_x = 0; dev_x < ntile_n_; dev_x++) {
								for (int dev_y = dev_x+s*ntile_n_; dev_y < num_devices; dev_y += 2*s*ntile_n_) {
									gpu_src = dev_y;
									gpu_dest = dev_y - s*ntile_n_; 

									tile_x = get_tile_size_row(dev_y/ntile_m_);
									if (s == 1 || (num_tile_cols%2 != 0 && dev_y/ntile_n_ == ntile_m_-1)) {
										cuda_exec(cudaMemcpyAsync(WRKSPACE_[gpu_dest], WRKSPACE_[gpu_src], block*sizeof(T)*ldWRK, cudaMemcpyDeviceToDevice, stream_[gpu_dest]));
									} else {
										cuda_exec(cudaMemcpyAsync(WRKSPACE_[gpu_dest], W[gpu_src], block*sizeof(T)*ldW, cudaMemcpyDeviceToDevice, stream_[gpu_dest]));
									}
									cuda_exec(cudaSetDevice(gpu_dest));	
									cuda_exec(cudaStreamSynchronize(stream_[gpu_dest]));
									cublasError = cublasTaxpy(handle_[gpu_dest], (pitchWRK[gpu_dest]/sizeof(T))*ldWRK, &one, WRKSPACE_[gpu_dest], 1, W[gpu_dest], 1);
									if(cublasError != CUBLAS_STATUS_SUCCESS) std::cout << "Error in Taxpy " << std::endl;
								}
							}
						}
					}

					//this->synchronizeAll();
					//stop = high_resolution_clock::now();
					//time_axpy += stop - start;					
	
					/* Finally, distribute the final result from the leading (first GPUs in the rows of the tiled H) to other GPUs in a row. 
 					 * This step is required in order to have all GPU updated for next call to computeHemm. */
					//int src_id, dest_id;

					//start = high_resolution_clock::now();

					/*if (next_ == NextOp::cAb) {
						
						for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {
							src_id = dev_x * ntile_n_;
							tile_x = get_tile_size_row(dev_x);
							for (int dev_y = 1; dev_y < ntile_n_; dev_y++) {
								dest_id = src_id + dev_y;
								cuda_exec(cudaMemcpy2DAsync(W[dest_id], pitchW[dest_id], W[src_id], pitchW[src_id], block*sizeof(T), tile_x, cudaMemcpyDeviceToDevice, stream_[dest_id]));
							}
						}
					} else {

						for (int dev_x = 0; dev_x < ntile_n_; dev_x++) {
							src_id = dev_x;
							tile_x = get_tile_size_col(dev_x);
							for (int dev_y = 1; dev_y < ntile_m_; dev_y++) {
								dest_id = src_id + ntile_n_*dev_y;
								cuda_exec(cudaMemcpy2DAsync(W[dest_id], pitchW[dest_id], W[src_id], pitchW[src_id], block*sizeof(T), tile_x, cudaMemcpyDeviceToDevice, stream_[dest_id]));
							}
						}
					}
					*/
					//this->synchronizeAll();
					//stop = high_resolution_clock::now();
					//time_redist += stop - start;
				}

				/* Collect and return the computed W from the GPUs to the host*/
				void return_W (T* buf_target, std::size_t ldBuf, std::size_t block) {

					/*  */
					int tile_x;
					int start_row;
					int src_gpu;

					//auto start = high_resolution_clock::now();

					if (next_ == NextOp::cAb) {
						for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {
							src_gpu = dev_x * ntile_n_;
							tile_x = get_tile_size_row(dev_x);
							start_row = dev_x * dim_tile_m_;

							cuda_exec(cudaSetDevice(src_gpu));
							cublasError = cublasGetMatrixAsync(tile_x, block, sizeof(T), W[src_gpu], ldW, &buf_target[start_row], ldBuf, stream_[src_gpu]);
							if(cublasError != CUBLAS_STATUS_SUCCESS) std::cout << "Error in cublasGetMatAsync W" << std::endl;
						}

					} else {
						for (int dev_x = 0; dev_x < ntile_n_; dev_x++) {
							src_gpu = dev_x;
							tile_x = get_tile_size_col(dev_x);
							start_row = dev_x * dim_tile_n_;

							cuda_exec(cudaSetDevice(src_gpu));
							cublasError = cublasGetMatrixAsync(tile_x, block, sizeof(T), W[src_gpu], ldW, &buf_target[start_row], ldBuf, stream_[src_gpu]);
							if(cublasError != CUBLAS_STATUS_SUCCESS) std::cout << "Error in cublasGetMatAsync W" << std::endl; 
						}

					}

					/* Synchronize all devices before return to the caller function*/
					//this->synchronizeAll();

					//auto stop = high_resolution_clock::now();
					//time_copy_W += stop - start;
				}

				/* Synchronize all devices */
				void synchronizeAll() {

					for (int i = 0; i < num_devices; i++) {
						cudaStreamSynchronize(stream_[i]);
					}
				}

				/* Switch pointers to per-device 2D arrays depending on the next operation (cAb or bAc) */
				void switch_pointers(){

					this->synchronizeAll();
					for(int gpu_id = 0; gpu_id < num_devices; gpu_id++) {
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

			private:

				/// Dimension of the input matrices H (m*n), W(m*maxBlock) and V(n*maxBlock)
				std::size_t n_;
				std::size_t m_;
				std::size_t maxBlock_;
				std::size_t max_dim_;

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

				/// Cublas handler for
				cublasHandle_t *handle_ = nullptr;
			
				/// Cuda streams
				cudaStream_t *stream_ = nullptr;

				/// Cublas error
				cublasStatus_t cublasError;

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
