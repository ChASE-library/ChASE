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

#include "algorithm/types.hpp"
#include "chase_mpi_properties.hpp"
#include "blas_cuda_wrapper.hpp"

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

				//mgpu_cudaHemm(cublasHandle_t handle, cudaStream_t stream, std::size_t m, std::size_t n, std::size_t maxBlock) :
				//handle_(handle), stream_(stream), m_(m), n_(n), maxBlock_(maxBlock) {
				mgpu_cudaHemm(std::size_t m, std::size_t n, std::size_t maxBlock) :
				m_(m), n_(n), maxBlock_(maxBlock) {
				
					/// Get number of available GPU devices
					cuda_exec(cudaGetDeviceCount(&num_devices));
					std::cout << "Running on " << num_devices << " devices!" << std::endl;

					/// Allocate array for device indices, handles and streams;
					devices_id = (int*) malloc(num_devices * sizeof(int));
					handle_ = (cublasHandle_t*) malloc(num_devices * sizeof(cublasHandle_t));
					stream_ = (cudaStream_t*) malloc(num_devices * sizeof(cudaStream_t));

					/// Populate list of devices, create handles and streams for each device
					for (int dev=0; dev<num_devices; dev++) {
						devices_id[dev] = dev;

						cuda_exec(cudaSetDevice(dev));
						cublasCreate(&handle_[dev]);
						cuda_exec(cudaStreamCreate(&stream_[dev]));
						cublasSetStream(handle_[dev], stream_[dev]);
					}

					/// Allocate arrays to hold pointers to memory allocations on each device for
					/// matrices H, B and IMT
					H_ = (T**) malloc(num_devices * sizeof(T*));
					B_ = (T**) malloc(num_devices * sizeof(T*));
					IMT_ = (T**) malloc(num_devices * sizeof(T*));
					WRKSPACE_ = (T**) malloc(num_devices * sizeof(T*));

					pitchB = (std::size_t*) malloc(num_devices * sizeof(std::size_t));
					pitchH = (std::size_t*) malloc(num_devices * sizeof(std::size_t));
					pitchIMT = (std::size_t*) malloc(num_devices * sizeof(std::size_t));
					pitchWRK = (std::size_t*) malloc(num_devices * sizeof(std::size_t));

					/* Compute number of tiles of matrix H
					 * The number of tiles depends on number of the available GPU devices */
					ntile_n_ = sqrt(num_devices);
					ntile_m_ = num_devices/ntile_n_;

					/* Compute tile dimensions */
					dim_tile_n_ = std::min(n_, (n_+ntile_n_-1)/ntile_n_);
					dim_tile_m_ = std::min(m_, (m_+ntile_m_-1)/ntile_m_);
					
					/* Set leading dimensions of the GPU arrays */
					//ldB = dim_tile_n_;
					//ldIMT = dim_tile_m_;
					ldWRK = std::max(dim_tile_m_, dim_tile_n_);
					ldB = ldIMT = ldWRK;
					ldH = dim_tile_m_;

					std::cout << "Number of tiles: "<<ntile_m_ << " x " << ntile_n_ << std::endl;
					std::cout << "Tile dimension: "<<dim_tile_m_ << " x " << dim_tile_n_ << std::endl;

					int dim_row = 0;
					int dim_col = 0;

					/* Pass through the tiles in row-major order (one-by-one tile row)
					 * compute dimension and allocate arrays asynchronously od devices */
					for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {

						/* Get the number of rows in the current tile */
						dim_row = get_tile_size_row(dev_x);

						for (int dev_y = 0; dev_y < ntile_n_; dev_y++) {

							/* Get the number of columns in the current tile */
							dim_col = get_tile_size_col(dev_y);

							/* Get the GPU id */
							int gpu_id = dev_x * ntile_n_ + dev_y;

							/* Set device */
							cuda_exec(cudaSetDevice(gpu_id));

							//* Allocate memories for H, IMT and B matrices */
							//cuda_exec(cudaMalloc((void**)&B_[gpu_id], ldB * maxBlock_ * sizeof(T)));
							//cuda_exec(cudaMalloc((void**)&IMT_[gpu_id], ldIMT * maxBlock_ * sizeof(T)));
							//cuda_exec(cudaMalloc((void**)&WRKSPACE_[gpu_id], ldWRK * maxBlock_ * sizeof(T)));
							//cuda_exec(cudaMalloc((void**)&H_[gpu_id], ldH * dim_col * sizeof(T)));
							cuda_exec(cudaMallocPitch((void**)&B_[gpu_id], &pitchB[gpu_id],  maxBlock_*sizeof(T), ldB));
							cuda_exec(cudaMallocPitch((void**)&IMT_[gpu_id], &pitchIMT[gpu_id], maxBlock_ * sizeof(T), ldIMT));
							cuda_exec(cudaMallocPitch((void**)&WRKSPACE_[gpu_id], &pitchWRK[gpu_id], maxBlock_ * sizeof(T), ldWRK));
							cuda_exec(cudaMallocPitch((void**)&H_[gpu_id], &pitchH[gpu_id], dim_col * sizeof(T), ldH));
						}
					}

					copied_ = false;
					next_ = NextOp::cAb;

					/* Start timing events */
					cuda_exec(cudaEventCreate(&start));
					cuda_exec(cudaEventCreate(&stop));
					time_copy_H = 0;
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
					free(devices_id);
					free(pitchB);
					free(pitchH);
					free(pitchIMT);
					free(pitchWRK);

					cudaEventDestroy(start);
					cudaEventDestroy(stop);
				}

				/* Divide given matrix H to block and distribute among GPUs */
				void distribute_H(T* orig_H, std::size_t ld_origH) {

					int dim_row, dim_col;
					int start_row, start_col;

					if( !copied_ ) {
						for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {
							dim_row = get_tile_size_row(dev_x);
							start_row = dev_x * dim_tile_m_;

							for(int dev_y = 0; dev_y < ntile_n_; dev_y++) {
								dim_col = get_tile_size_col(dev_y);
								start_col = dev_y * dim_tile_n_;

								int dev_id = dev_x * ntile_n_ + dev_y;
								cuda_exec(cudaSetDevice(dev_id));
								//cuda_exec(cudaMemcpy2DAsync(H_[dev_id], Hpitch, &orig_H[start_row * n_ + start_col], n_*sizeof(T), dim_col * sizeof(T), dim_row, cudaMemcpyHostToDevice, stream_[dev_id]));
								//cublasError = cublasSetMatrixAsync(dim_row, dim_col, sizeof(T), &orig_H[start_row*n_ + start_col], ld_origH, H_[dev_id], ldH, stream_[dev_id]);
								cublasError = cublasSetMatrixAsync(dim_row, dim_col, sizeof(T), &orig_H[start_col * ld_origH + start_row], ld_origH, H_[dev_id], ldH, stream_[dev_id]);
								if(cublasError != CUBLAS_STATUS_SUCCESS) std::cout << "Error in cublasSetMatrixAsync H" << std::endl;
							}
						}
						copied_ = true;
					}
					else {
						for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {
							dim_row = get_tile_size_row(dev_x);
							start_row = dev_x * dim_tile_m_;

							for(int dev_y = 0; dev_y < ntile_n_; dev_y++) {
								dim_col = get_tile_size_col(dev_y);
								start_col = dev_y * dim_tile_n_;

								int dev_id = dev_x * ntile_n_ + dev_y;
								cuda_exec(cudaSetDevice(dev_id));
								//cuda_exec(cudaMemcpy2DAsync(H_[dev_id], Hpitch, &orig_H[start_row * n_ + start_col], n_*sizeof(T), dim_col * sizeof(T), dim_row, cudaMemcpyHostToDevice, stream_[dev_id]));
								//cublasError = cublasSetMatrixAsync(dim_row, dim_col, sizeof(T), &orig_H[start_row*n_ + start_col], ld_origH, H_[dev_id], ldH, stream_[dev_id]);
								cublasError = cublasSetMatrixAsync(dim_row, dim_col, sizeof(T), &orig_H[start_col * ld_origH + start_row], ld_origH, H_[dev_id], ldH, stream_[dev_id]);
								if(cublasError != CUBLAS_STATUS_SUCCESS) std::cout << "Error in cublasSetMatrixAsync H" << std::endl;
							}
						}

					}
				}

				/* Divide given tall-skinny matrices V/W into panels and distributed them among GPUs */
				void distribute_V (T* buf_init, std::size_t ldBuf, std::size_t block) {

					/// Dimension of the block and start index
					int dim_row;
					int start_row;

					if (next_ == NextOp::cAb) {

						/* Divide buf_init input tile ntile_n_ tiles and distribute to GPU such as
 						 * that tile i goes to the GPUs with GPU_ID % ntile_n_ == i
 						 */ 
						for (int i=0; i<ntile_n_; i++) {

							/* buf_init is divided into row-panels which dimension and starting index 
 							 * correspond to the dimension of the tiled matrix H by columns
							 */
							dim_row = get_tile_size_col(i);
							start_row = i * dim_tile_n_;

							for (int dev = i; dev < num_devices; dev += ntile_n_) {
								cuda_exec(cudaSetDevice(dev));
								//cuda_exec(cudaMemcpy2DAsync(B_[dev], Bpitch, &buf_init[start_row*maxBlock_], maxBlock_*sizeof(T), block*sizeof(T), dim_row, cudaMemcpyHostToDevice, stream_[dev]));
								//cublasError = cublasSetMatrixAsync(dim_row, block, sizeof(T), &buf_init[start_row*maxBlock_], ldBuf, B_[dev], ldB, stream_[dev]);
								cublasError = cublasSetMatrixAsync(dim_row, block, sizeof(T), &buf_init[start_row], ldBuf, B_[dev], ldB, stream_[dev]);
								if(cublasError != CUBLAS_STATUS_SUCCESS) std::cout << "Error in cublasSetMatrixAsync V/W" << std::endl;

							}
						}
					//cuda_exec(cudaMemcpyAsync(B_, buf_init, block * k * sizeof(T), 
					//						  cudaMemcpyHostToDevice, stream_));
					}
					else {

						/* Divide buf_init input tile ntile_m_ tiles and distribute to GPU such as
 						 * that tile i goes to the GPUs with GPU_ID % ntile_m_ == i
 						 */ 
						for (int i = 0; i < ntile_m_; i++) {

							/* buf_init is divided into row-panels which dimension and starting index 
 							 * correspond to the dimension of the tiled matrix H by columns
							 */
							dim_row = get_tile_size_row(i);
							start_row = i * dim_tile_m_;
							
							int start_dev_id = i * ntile_n_;
							for (int dev = start_dev_id; dev < start_dev_id + ntile_n_; dev++) {
								cuda_exec(cudaSetDevice(dev));
								//cuda_exec(cudaMemcpy2DAsync(B_[dev], Bpitch, &buf_init[start_row*maxBlock_], maxBlock_*sizeof(T), block*sizeof(T), dim_row, cudaMemcpyHostToDevice, stream_[dev]));
								//cublasError = cublasSetMatrixAsync(dim_row, block, sizeof(T), &buf_init[start_row*maxBlock_], ldBuf, B_[dev], ldB, stream_[dev]);
								cublasError = cublasSetMatrixAsync(dim_row, block, sizeof(T), &buf_init[start_row], ldBuf, B_[dev], ldB, stream_[dev]);
								if(cublasError != CUBLAS_STATUS_SUCCESS) std::cout << "Error in cublasSetMatrixAsync V/W" << std::endl;
							}
						}
					}
				}

				/* Compute Hemm */
				//void computeHemm(T* buf_init, T* buf_target, std::size_t m, std::size_t n, std::size_t k, std::size_t block, T alpha, T beta, cublasOperation_t transa) {
				//void computeHemm(std::size_t m, std::size_t n, std::size_t k, T alpha, T beta) {
				void computeHemm(std::size_t block, T alpha, T beta) {

					std::cout << "Computing Hemm..." << std::endl;
					std::size_t m, n, k;
					std::size_t tile_dim_row, tile_dim_col;
					std::size_t num_tile_rows, num_tile_cols;
					T zero = 0;
					T one = 1;
					bool leading_gpu = false;
					cublasOperation_t transa;

					if (next_ == NextOp::bAc) {

					//cublasTgemm(handle_, transa, CUBLAS_OP_N, m, n, k, &alpha, H_, m_, B_, k,
					//			&beta, IMT_, m);

						transa = CUBLAS_OP_C;
						num_tile_rows = ntile_n_;
						num_tile_cols = ntile_m_;
						//next_ = NextOp::cAb;
					} else {
						transa = CUBLAS_OP_N;
						num_tile_rows = ntile_m_;
						num_tile_cols = ntile_n_;
						//next_ = NextOp::bAc;
					}

					for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {
						
						tile_dim_row = get_tile_size_row(dev_x);
						for (int dev_y = 0; dev_y < ntile_n_; dev_y++) {
						
							tile_dim_col = get_tile_size_col(dev_y);

							if (next_ == NextOp::cAb) {
								m = tile_dim_row;
								n = block;
								k = tile_dim_col;
								if(dev_y == 0) {
									leading_gpu = true;
								} else {
									leading_gpu = false;
								}
							} else {
								m = tile_dim_col;
								n = block;
								k = tile_dim_row;
								if(dev_x == 0) {
									leading_gpu = true;
								} else {
									leading_gpu = false;
								}
							}
							int dev_id = dev_x * ntile_n_ + dev_y;
		
							cuda_exec(cudaSetDevice(dev_id));
							if (leading_gpu) {
								cublasTgemm(handle_[dev_id], transa, CUBLAS_OP_N, m, n, k, &alpha, H_[dev_id], ldH, B_[dev_id], ldB,
									&beta, IMT_[dev_id], ldIMT);
							} else {
								cublasTgemm(handle_[dev_id], transa, CUBLAS_OP_N, m, n, k, &alpha, H_[dev_id], ldH, B_[dev_id], ldB,
									&zero, WRKSPACE_[dev_id], ldWRK);
							}
						}
					}
					
					/* Synchronize all GPUs */
					this->synchronizeAll();

					/* Compute the final solution from partial per-GPU solutions */
					/* */
					for (int s = 1; s < num_tile_cols; s <<= 1) {

						for (int dev = s; dev < num_devices; dev += 2*s) {
							tile_dim_row = get_tile_size_row(dev/ntile_n_);

							if (s == 1) {
								//cuda_exec(cudaMemcpy2DAsync(WRKSPACE_[dev-s], pitchWRK[dev-s], WRKSPACE_[dev], pitchWRK[dev], block*sizeof(T), tile_dim_row, cudaMemcpyDeviceToDevice, stream_[dev-s]));
								//cuda_exec(cudaMemcpy2DAsync(WRKSPACE_[dev-s], pitchWRK[dev-s], WRKSPACE_[dev], pitchWRK[dev], maxBlock_*sizeof(T), ldWRK, cudaMemcpyDeviceToDevice, stream_[dev-s]));
								cuda_exec(cudaMemcpyAsync(WRKSPACE_[dev-s], WRKSPACE_[dev], block*sizeof(T)*ldWRK, cudaMemcpyDeviceToDevice, stream_[dev-s]));
							} else {
								//cuda_exec(cudaMemcpy2DAsync(WRKSPACE_[dev-s], pitchWRK[dev-s], IMT_[dev], pitchIMT[dev], block*sizeof(T), tile_dim_row, cudaMemcpyDeviceToDevice, stream_[dev-s]));
								cuda_exec(cudaMemcpyAsync(WRKSPACE_[dev-s], IMT_[dev], block*sizeof(T)*ldIMT, cudaMemcpyDeviceToDevice, stream_[dev-s]));
							}

							// TODO: Might be critical since WRKSPACE has paddings. Rows might be longer than maxBlock_ and Taxpy does not catch all the elements.
							// Straight forward solution is to compute Taxpy on the entire WRKSPACE (ldWRK * maxBlock_)
							cuda_exec(cudaSetDevice(dev-s));	
							cuda_exec(cudaStreamSynchronize(stream_[dev-s]));
							//cublasError = cublasTaxpy(handle_[dev-s], block*tile_dim_row, &one, WRKSPACE_[dev-s], 1, IMT_[dev-s], 1);
							cublasError = cublasTaxpy(handle_[dev-s], (pitchWRK[dev-s]/sizeof(T))*ldWRK, &one, WRKSPACE_[dev-s], 1, IMT_[dev-s], 1);
							if(cublasError != CUBLAS_STATUS_SUCCESS) std::cout << "Error in Taxpy " << std::endl;
						}
						
					}

					/* Disturibute the results from the leading GPUs to other GPUs in a row */
					int src_id, dest_id;

					if (next_ == NextOp::cAb) {
						
						for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {
							src_id = dev_x * ntile_n_;
							tile_dim_row = get_tile_size_row(dev_x);
							for (int dev_y = 1; dev_y < ntile_n_; dev_y++) {
								dest_id = src_id + dev_y;
								//cuda_exec(cudaMemcpy2DAsync(IMT_[dest_id], IMTpitch, IMT_[src_id], IMTpitch, block*sizeof(T), tile_dim_row, cudaMemcpyDeviceToDevice, stream_[dest_id]));
								//cuda_exec(cudaMemcpyAsync(IMT_[dest_id], IMT_[src_id], block*sizeof(T) * tile_dim_row, cudaMemcpyDeviceToDevice, stream_[dest_id]));
								cuda_exec(cudaMemcpy2DAsync(IMT_[dest_id], pitchIMT[dest_id], IMT_[src_id], pitchIMT[src_id], block*sizeof(T), tile_dim_row, cudaMemcpyDeviceToDevice, stream_[dest_id]));
							}
						}
					} else {

						for (int dev_x = 0; dev_x < ntile_n_; dev_x++) {
							src_id = dev_x;
							tile_dim_row = get_tile_size_col(dev_x);
							for (int dev_y = 1; dev_y < ntile_m_; dev_y++) {
								dest_id = src_id + ntile_n_*dev_y;
								//cuda_exec(cudaMemcpy2DAsync(IMT_[dest_id], IMTpitch, IMT_[src_id], IMTpitch, block*sizeof(T), tile_dim_row, cudaMemcpyDeviceToDevice, stream_[dest_id]));
								//cuda_exec(cudaMemcpyAsync(IMT_[dest_id], IMT_[src_id], block*sizeof(T) * tile_dim_row, cudaMemcpyDeviceToDevice, stream_[dest_id]));
								cuda_exec(cudaMemcpy2DAsync(IMT_[dest_id], pitchIMT[dest_id], IMT_[src_id], pitchIMT[src_id], block*sizeof(T), tile_dim_row, cudaMemcpyDeviceToDevice, stream_[dest_id]));
							}
						}
					}
				}

				/* Collect the computed V/W from the GPUs */
				void return_V (T* buf_target, std::size_t ldBuf, std::size_t block) {

					//cuda_exec(cudaMemcpyAsync(buf_target, IMT_, m * block * sizeof(T),
					//		  				  cudaMemcpyDeviceToHost, stream_));
					
					/*  */
					int tile_dim_row;
					int start_row;
					int src_gpu;

					if (next_ == NextOp::cAb) {
						for (int dev_x = 0; dev_x < ntile_m_; dev_x++) {
							src_gpu = dev_x * ntile_n_;
							tile_dim_row = get_tile_size_row(dev_x);
							start_row = dev_x * dim_tile_m_;

							cuda_exec(cudaSetDevice(src_gpu));
							//cuda_exec(cudaMemcpy2DAsync(&buf_target[start_row * maxBlock_], maxBlock_*sizeof(T), IMT_[src_gpu], IMTpitch, block*sizeof(T), tile_dim_row, cudaMemcpyDeviceToHost, stream_[src_gpu]));
							//cublasError = cublasGetMatrixAsync(tile_dim_row, block, sizeof(T), IMT_[src_gpu], ldIMT, &buf_target[start_row * maxBlock_], ldBuf, stream_[src_gpu]);
							cublasError = cublasGetMatrixAsync(tile_dim_row, block, sizeof(T), IMT_[src_gpu], ldIMT, &buf_target[start_row], ldBuf, stream_[src_gpu]);
							if(cublasError != CUBLAS_STATUS_SUCCESS) std::cout << "Error in cublasGetMatAsync W" << std::endl;
						}

					} else {
						for (int dev_x = 0; dev_x < ntile_n_; dev_x++) {
							src_gpu = dev_x;
							tile_dim_row = get_tile_size_col(dev_x);
							start_row = dev_x * dim_tile_n_;

							cuda_exec(cudaSetDevice(src_gpu));
							//cuda_exec(cudaMemcpy2DAsync(&buf_target[start_row * maxBlock_], maxBlock_*sizeof(T), IMT_[src_gpu], IMTpitch, block*sizeof(T), tile_dim_row, cudaMemcpyDeviceToHost, stream_[src_gpu]));
							//cublasError = cublasGetMatrixAsync(tile_dim_row, block, sizeof(T), IMT_[src_gpu], ldIMT, &buf_target[start_row * maxBlock_], ldBuf, stream_[src_gpu]);
							cublasError = cublasGetMatrixAsync(tile_dim_row, block, sizeof(T), IMT_[src_gpu], ldIMT, &buf_target[start_row], ldBuf, stream_[src_gpu]);
							if(cublasError != CUBLAS_STATUS_SUCCESS) std::cout << "Error in cublasGetMatAsync W" << std::endl; 
						}

					}

					/* Synchronize all devices before return to the caller function*/
					this->synchronizeAll();
				}

				/// Synchronize all streams
				void synchronizeAll() {

					for (int i = 0; i < num_devices; i++) {
						cudaStreamSynchronize(stream_[i]);
					}
				}

				/// Swap V and W (switch between C = A B + C and C = A^T B + C 
				void swap_VW() {
					if ( next_ == NextOp::bAc) {
						next_ = NextOp::cAb;
					} else {
						next_ = NextOp::bAc;
					}
				}

			private:

				std::size_t n_;
				std::size_t m_;
				std::size_t maxBlock_;
				std::size_t max_dim_;

				T** B_ = nullptr;
				T** IMT_ = nullptr;
				T** H_ = nullptr;
				T** WRKSPACE_ = nullptr;

				/// Leading dimensions
				std::size_t ldB;
				std::size_t ldIMT;
				std::size_t ldWRK;
				std::size_t ldH;
				
				/// Pitched values
				std::size_t *pitchB = nullptr;
				std::size_t *pitchIMT = nullptr;
				std::size_t *pitchWRK = nullptr;
				std::size_t *pitchH = nullptr;
				
				bool copied_;

				enum NextOp { cAb, bAc };

				/// Keep the record of the next operation type (c = A b + c or c = b A^T +c )
				NextOp next_;

				/// List of GPU devices
				int num_devices;
				int *devices_id = nullptr;

				/// Number of tiles in m/n direction
				int ntile_m_;
				int ntile_n_;

				/// Size of the tiles
				int dim_tile_m_;
				int dim_tile_n_;

  				/// Timing values
  				cudaEvent_t start, stop;
				float time_copy_H;

				/// Cublas handler for
				cublasHandle_t *handle_ = nullptr;
			
				/// Cuda streams
				cudaStream_t *stream_ = nullptr;

				/// Cublas error
				cublasStatus_t cublasError;

				/// Return x-dimension of tile in m-direction
				int get_tile_size_row (int tile_position) {
					if (tile_position + 1 == ntile_m_) {
						return m_ - tile_position * dim_tile_m_;
					} else {
						return dim_tile_m_;
					}
				}
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
