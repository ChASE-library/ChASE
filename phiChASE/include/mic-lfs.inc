#!/bin/bash
# Template for Sampler configuration files.
# This file will be sourced in make.sh at built time.

# Environment variables are used if available but overwritten by the variables
# in this file.
# Variables commented out have default values, which are used if they are set
# neither here nor in the environment.
# items that are only relevant for the GUI are marked [GUI]


# SYSTEM, COMPILERS and LIBRARIES

# Name of the BLAS library (e.g., "MKL", "OpenBLAS", "BLIS")
#
BLAS_NAME=MKL

# Name of the System (e.g., "local", "cluster", "E5450")
#
SYSTEM_NAME="mic-cluster"

# Name of the Sampler
#
#NAME=#{SYSTEM_NAME}_${BLAS_NAME}

# Number of cores
#
NT_MAX=244

# C compiler (default: "gcc")
#
CC=icc

# C compiler flags (default: "")
#
CFLAGS=-mmic

# C++ compiler (default: "g++")
#
CXX=icpc

# C++ compiler flags (default: "")
#
CXXFLAGS=-mmic

# Linker flags (e.g., "-lopenblas")
#
LINK_FLAGS="-mmic -mkl"

# Include flags (e.g. "-I${PAPI_ROOT}/include)
#
#INCLUDE_FLAGS=


# KERNELS

# Header files for kernels
# Will be processed by C preprocessor.
# Should only contain function declarations.
#
#KERNEL_HEADERS="headers/blas.h headers/lapack.h"


# PAPI

# If PAPI is installed, the following information can be optained from
# papi_avail.

# Number of counters the CPU supports simmultaneously (default: 0)
# 0: disable PAPI support entirely.
# Listed as "Number Hardware Counters" in papi_avail
#
#PAPI_COUNTERS_MAX=0

# Whitespac separated list of counter names available on the system [GUI]
# e.g. "PAPI_L1_DCM PAPI_L1_ICM PAPI_L2_DCM"
#
#PAPI_COUNTERS_AVAIL=


# BACKEND

# Name of the backend for job submissions [GUI]
# Requires a match script in GUI/src/backends/${BACKEND}.py
# local: jobs run on the system runnig the GUI, i.e., the frontend.
#
BACKEND="lsf"

# Additional script lines required before running the Sampler [GUI]
# {nt} will be replaced by the number of threads.
# e.g.: "export OMP_NUM_THREADS={nt}"
#
BACKEND_HEADER=" 
#BSUB -W 4:00
#BSUB -M 1000
#BSUB -x
#BSUB -n 1
#BSUB -P aices
#BSUB -a phi
#BSUB -Jd \"leo=1;{nt}\"
#BSUB -R \"select[hpcwork]\"
ssh cluster-phi-mic0 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/Compiler/14.0/3.174/rwthlnk/mkl/lib/mic/ MKL_NUM_THREADS={nt} KMP_AFFINITY=scatter \\"

# Additional options passed to submit() [GUI]
# Format: python dictionary representation
# (for future use)
#
#BACKEND_OPTIONS="{}"


# CPU INFORMATION

# These are required to e.g. compute performance and efficiencies in the GUI

# Full CPU model name [GUI]
# e.g. from /proc/cpuinfo
#
CPU_MODEL=XeonPhi

# CPU Frequency in MHz [GUI]
#
FREQUENCY_MHZ=1005

# Maximum number of double preicion floating point operations issued per cycle
# [GUI]
#
DFLOPS_PER_CYCLE=16

# Maxiumum number of single precision floating point operations issued per cycle
# [GUI
#
#SFLOPS_PER_CYCLE=$((2 * DFLOPS_PER_CYCLE))
