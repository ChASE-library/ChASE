# Modern Makefile for ChASE using pkg-config
# This replaces manual configuration with automatic detection

C = mpicc
CXX = mpicxx
FORTRAN = mpif90

CFLAGS = -fopenmp -MMD -lm
CXXFLAGS = -fopenmp -MMD -lm
FFLAGS = -fopenmp -MMD -cpp -lm

# ============================================================================
# NEW: Use pkg-config for automatic configuration
# ============================================================================

# Get all ChASE configuration automatically
CHASE_CFLAGS := $(shell pkg-config --cflags chase)
CHASE_LIBS := $(shell pkg-config --libs chase)

# Query ChASE features (much cleaner than using nm!)
CHASE_HAS_MPI := $(shell pkg-config --variable=chase_has_mpi chase)
CHASE_HAS_NCCL := $(shell pkg-config --variable=chase_has_nccl chase)
CHASE_HAS_SCALAPACK := $(shell pkg-config --variable=chase_has_scalapack chase)

# Get ChASE installation prefix if needed for libraries
CHASE_PREFIX := $(shell pkg-config --variable=prefix chase)
CHASE_LIBDIR := $(shell pkg-config --variable=libdir chase)

# ============================================================================
# Build library dependencies based on what ChASE has
# ============================================================================

LIBS_BLASLAPACK = -lopenblas -lgfortran -lscalapack -lstdc++

# Conditionally add GPU libraries if ChASE was built with them
ifeq ($(CHASE_HAS_NCCL),ON)
    LIBS_CUDA = -lcublas -lcusolver -lcudart -lcurand
    LIBS_NCCL = -lnccl
    LIBS_CHASE_CUDA = $(CHASE_LIBDIR)/libchase_cuda_kernels.a
    GPU_LIBS = $(LIBS_CUDA) $(LIBS_NCCL) -Wl,--whole-archive $(LIBS_CHASE_CUDA) -Wl,--no-whole-archive
else
    GPU_LIBS =
endif

# C/Fortran interface libraries
LIBS_CHASE_C = $(CHASE_LIBDIR)/libchase_c.a
LIBS_CHASE_F = $(CHASE_LIBDIR)/libchase_f.a

# Combined libraries
LIBS = $(LIBS_BLASLAPACK) $(GPU_LIBS)

# ============================================================================
# Source files and executables
# ============================================================================

src = chase-app.cpp
exe = chase-app

src_c = chase-c.c
exe_c = chase-c

src_f = chase-f.f90
exe_f = chase-f

all: ${exe} ${exe_c} ${exe_f}

.SUFFIXES:

# ============================================================================
# Build rules using pkg-config
# ============================================================================

$(exe): $(src)
	@echo "Building C++ example with ChASE..."
	@echo "  ChASE flags: $(CHASE_CFLAGS)"
	@echo "  Features: MPI=$(CHASE_HAS_MPI) NCCL=$(CHASE_HAS_NCCL) ScaLAPACK=$(CHASE_HAS_SCALAPACK)"
	$(CXX) $(CXXFLAGS) $(CHASE_CFLAGS) -o $@ $< $(LIBS)

$(exe_c): $(src_c)
	@echo "Building C example with ChASE..."
	$(C) $(CFLAGS) $(CHASE_CFLAGS) -o $@ $< \
		-L$(CHASE_LIBDIR) -lchase_c \
		$(LIBS_BLASLAPACK) \
		$(GPU_LIBS) \
		-lm

$(exe_f): $(src_f)
	@echo "Building Fortran example with ChASE..."
	$(FORTRAN) $(FFLAGS) $(CHASE_CFLAGS) -o $@ $< \
		-L$(CHASE_LIBDIR) -lchase_c -lchase_f \
		$(LIBS_BLASLAPACK) \
		$(GPU_LIBS) \
		-lm

clean:
	-rm -f $(exe) $(exe_c) $(exe_f) *.o *.d

# ============================================================================
# Utility targets
# ============================================================================

info:
	@echo "ChASE Configuration (from pkg-config):"
	@echo "  Installation prefix: $(CHASE_PREFIX)"
	@echo "  Library directory:   $(CHASE_LIBDIR)"
	@echo "  Compile flags:       $(CHASE_CFLAGS)"
	@echo "  Link flags:          $(CHASE_LIBS)"
	@echo ""
	@echo "Feature Detection:"
	@echo "  MPI support:       $(CHASE_HAS_MPI)"
	@echo "  NCCL support:      $(CHASE_HAS_NCCL)"
	@echo "  ScaLAPACK support: $(CHASE_HAS_SCALAPACK)"

check-pkg-config:
	@pkg-config --exists chase && echo "✓ ChASE found via pkg-config" || \
		(echo "✗ ChASE not found! Set PKG_CONFIG_PATH" && exit 1)

.PHONY: all clean info check-pkg-config
