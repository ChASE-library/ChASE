CXX = mpicxx

CXXFLAGS = -fopenmp -MMD #-Wl,--whole-archive 

INCLUDE_DIR = ${ChASEROOT}/include

LIBS_BLASLAPACK = -lopenblas -lgfortran -lscalapack -lstdc++ 

LIBS_CUDA = -lcublas -lcusolver -lcudart -lcurand -lnccl
LIBS_CHASE_CUDA = ${ChASEROOT}/lib64/libchase_cuda_kernels.a

DEFINES_CPU = -DCHASE_OUTPUT -DUSE_NVTX -DHAS_SCALAPACK
DEFINES_GPU = -DHAS_CUDA -DHAS_NCCL -DUSE_GPU

USE_GPU ?= 0  # Default is CPU

# Set the appropriate defines based on USE_GPU flag
ifeq ($(USE_GPU), 1)
    DEFINES = ${DEFINES_CPU} ${DEFINES_GPU}
    LIBS = ${LIBS_BLASLAPACK} ${LIBS_CUDA} -Wl,--whole-archive ${LIBS_CHASE_CUDA} -Wl,--no-whole-archive
else
    DEFINES = ${DEFINES_CPU}
    LIBS = ${LIBS_BLASLAPACK}
endif

src = chase-app.cpp
exe = chase-app

all: ${exe}

.SUFFIXES:

$(exe): $(src)
	${CXX} ${CXXFLAGS} ${DEFINES} ${LIBS} -I${INCLUDE_DIR} -o $@ $<

clean:
	-rm -f $(exe) *.o
	-rm -f $(exe) *.d
