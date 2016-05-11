# -*- mode: Makefile;-*-
# Ensure that the obj/ folder exists
MKDIR_P = mkdir -p
AR = /usr/bin/ar
ARFLAGS = rcs
# Build the primitive CUDA filter
# Requires CUDA, and does not deliver good performance for large matrices
#BUILD_CUDA=1

CFLAGS+=-O3 -std=c++11
LDFLAGS+=-mkl

CXX=icpc
CPPFLAGS+=-qopenmp

CUDACXX=nvcc
CUDAFLAGS+=

# Only required when BUILD_CUDA is set
CUDA_INCLUDE_DIR=${CUDA_ROOT}/include
CUDA_LIBS_DIR=${CUDA_ROOT}/lib64

#Boost is required for the main.cpp driver routine
BOOST_INCLUDE_DIR=${BOOST_ROOT}/include
BOOST_LIBS_DIR=${BOOST_ROOT}/lib
