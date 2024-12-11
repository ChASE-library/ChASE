C = mpicc
CXX = mpicxx
FORTRAN = mpif90

CFLAGS = -fopenmp -MMD -lm
CXXFLAGS = -fopenmp -MMD -lm
FFLAGS = -fopenmp -MMD -cpp -lm

INCLUDE_DIR = ${ChASEROOT}/include

LIBS_BLASLAPACK = -lopenblas -lgfortran -lscalapack -lstdc++ 

LIBS_CUDA = -lcublas -lcusolver -lcudart -lcurand 
LIBS_NCCL = -lnccl
LIBS_CHASE_CUDA = ${ChASEROOT}/lib64/libchase_cuda_kernels.a
LIBS_CHASE_C = ${ChASEROOT}/lib64/libchase_c.a
LIBS_CHASE_F = ${ChASEROOT}/lib64/libchase_f.a

DEFINES_CPU = -DCHASE_OUTPUT -DUSE_NVTX -DHAS_SCALAPACK
DEFINES_GPU = -DHAS_CUDA -DHAS_NCCL -DUSE_GPU

# Check if blockcyclic is present in the library (at least one occurrence)
# this is only for C/Fortran interface
ifeq ($(shell nm ${ChASEROOT}/lib64/libchase_c.a | grep -c blockcyclic), 0)
    # If no blockcyclic is found, do nothing
else
    # If at least one blockcyclic is found, append -DINTERFACE_BLOCK_CYCLIC to DEFINES
    DEFINES_CPU := $(DEFINES_CPU) -DINTERFACE_BLOCK_CYCLIC
endif

## for the cpp example
## This example assume NCCL is available if GPU is used.
USE_GPU := $(shell nm ${ChASEROOT}/lib64/libchase_c.a | grep -E "cuda_mpi|cuda_nccl" > /dev/null && echo 1 || echo 0)

# Set the appropriate defines based on USE_GPU flag
ifeq ($(USE_GPU), 1)
    DEFINES = ${DEFINES_CPU} ${DEFINES_GPU}
    LIBS = ${LIBS_BLASLAPACK} ${LIBS_CUDA} ${LIBS_NCCL} -Wl,--whole-archive ${LIBS_CHASE_CUDA} -Wl,--no-whole-archive
else
    DEFINES = ${DEFINES_CPU}
    LIBS = ${LIBS_BLASLAPACK}
endif

src = chase-app.cpp
exe = chase-app

src_c = chase-c.c
exe_c = chase-c

src_f = chase-f.f90
exe_f = chase-f

all: ${exe} ${exe_c} ${exe_f}

.SUFFIXES:

$(exe): $(src)
	${CXX} ${CXXFLAGS} ${DEFINES} ${LIBS} -I${INCLUDE_DIR} -o $@ $<

$(exe_c): $(src_c)
	${C} ${CFLAGS} ${DEFINES} -I${INCLUDE_DIR} chase-c.c \
		-L${ChASEROOT}/lib64 -lchase_c \
		${LIBS_BLASLAPACK} \
		${LIBS_CUDA} \
		${LIBS_NCCL} \
		-Wl,--whole-archive ${LIBS_CHASE_CUDA} \
		-Wl,--no-whole-archive \
		-lm \
		-o $@

$(exe_f): $(src_f)
	${FORTRAN} ${FFLAGS} ${DEFINES} -I${INCLUDE_DIR} chase-f.f90 \
		-L${ChASEROOT}/lib64 -lchase_c -lchase_f \
		${LIBS_BLASLAPACK} \
		${LIBS_CUDA} \
		${LIBS_NCCL} \
		-Wl,--whole-archive ${LIBS_CHASE_CUDA} \
		-Wl,--no-whole-archive \
		-lm \
		-o $@

clean:
	-rm -f $(exe) ${exe_c} ${exe_f} *.o *.d 