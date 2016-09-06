include make.inc
################################################################################
CUDA_FILE_DIR=src
CPP_FILE_DIR=src
HEADER_FILES := $(wildcard include/*.h)
CPP_FILES := $(wildcard ${CPP_FILE_DIR}/*.cpp)

ifdef BUILD_CUDA
CUDA_FILES := $(wildcard ${CUDA_FILE_DIR}/*.cu)
CFLAGS+=-DCHASE_BUILD_CUDA
CFLAGS_CUDA=-I${CUDA_INCLUDE_DIR}
LDFLAGS_CUDA=-L${CUDA_LIBS_DIR} -lcudart -lcublas -lnvToolsExt
else
CUDA_FILES =
CFLAGS_CUDA=
LDFLAGS_CUDA=
endif

CFLAGS_BOOST:=-I${BOOST_INCLUDE_DIR}
LDFLAGS_BOOST=-L${BOOST_LIBS_DIR} -lboost_program_options -lboost_serialization

CFLAGS+=${CFLAGS_BOOST} ${CFLAGS_CUDA}
LDFLAGS+=${LDFLAGS_BOOST} ${LDFLAGS_CUDA}

CPPFLAGS+=${CFLAGS} ${FLAG_OPENMP}
CUDAFLAGS+=${CFLAGS}

CPP_OBJS := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
CUDA_OBJS := $(addprefix obj/,$(notdir $(CUDA_FILES:.cu=.o)))
OBJS=${CUDA_OBJS} ${CPP_OBJS}

all: lib

${OBJS}: ${HEADER_FILES}

lib: obj_dir obj/chase.o obj/filter.o obj/lanczos.o obj/timings.o
	${AR} ${ARFLAGS} libchase.a  obj/chase.o obj/filter.o obj/lanczos.o obj/timings.o

main: obj_dir ${OBJS}
	${CXX} ${OBJS} ${LDFLAGS} -o $@.x

obj/%.o: ${CPP_FILE_DIR}/%.cpp
	${CXX} ${CPPFLAGS} -c $< -o $@

obj/%.o: ${CUDA_FILE_DIR}/%.cu
	${CUDACXX} ${CUDAFLAGS}  -c $< -o $@

.PHONY: obj_dir clean test

obj_dir:
	${MKDIR_P} obj/

clean:
	rm -f obj/* main.x libchase.a
