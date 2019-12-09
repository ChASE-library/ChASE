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
				mgpu_cudaHemm();

				/* Remove local variabls */
				~mgpu_cudaHemm();

				/* Divide given matrix H to block and distribute among GPUs */
				void distributeH();

				/* Divide given tall-skinny matrices V/W into panels and distributed them among GPUs */
				void distributeV();

				/* Compute Hemm */
				void computeHemm();

				/* Collect the computed V/W from the GPUs */
				void returnV();

			private:

				/// List of GPU devices
				int *gpu_devices = nullptr;

				/// Cublas handler for
				cublasHandle_t handle;
			
				/// Cuda streams
				cudaStream_t stream;

		};
	}  // namespace matrixfree
}  // namespace chase
