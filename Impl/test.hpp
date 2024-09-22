#pragma once

//#define DATA_TYPE double
//#define ALGORITHM_PREFIX test::Impl::CPU::SHMCPU<DATA_TYPE>
//#include "MPICPUImpl.hpp"
//#include "DistributionMode.hpp"
//#define ALGORITHM_PREFIX test::Impl::MPI::CPU::MPICPU<DATA_TYPE, test::DistributionMode::BlockCyclic>
//#define DATA_TYPES float, double, std::complex<double>

#ifdef ENABLE_SHMCPU
#include "ShmCPUImpl.hpp"
#define ALGORITHM_PREFIX test::Impl::CPU::SHMCPU
#else
#include "ShmGPUImpl.hpp"
#define ALGORITHM_PREFIX test::Impl::GPU::SHMGPU
#endif

#include "MPICPUImpl.hpp"

