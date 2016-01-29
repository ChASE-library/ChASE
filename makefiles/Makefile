#######################
#######################
####################### WORK IN PROGRESS! NOT USABLE!
#######################
#######################

# include the conf files

include phi.conf
include cuda.conf
include elemental.conf

### VARIABLES DEFINITION

CC=icpc

EXEC_PATH = "../example drivers/exec"
DRIVER_PATH = "../example drivers/"

MIC_SRC_PATH = ../phiChASE/src
MIC_INC_PATH = ../phiChASE/include
MIC_OBJ_PATH = ../phiChASE/obj
MIC_DRV_PATH = "../example drivers/phiChASE/"
MIC_SRCS = $(MIC_SRC_PATH)/chfsiP.cpp $(MIC_SRC_PATH)/lanczos.cpp $(MIC_SRC_PATH)/filter.cpp
MIC_INCS = $(MIC_INC_PATH)/chfsi.h $(MIC_INC_PATH)/lanczos.h $(MIC_INC_PATH)/filter.h
MIC_OBJS = $(MIC_OBJ_PATH)/phichase.o
### MIC_OBJS = $(addprefix ../phiChASE/obj/,$(notdir $(MIC_SRCS:.cpp=.o))) 
# # ../phiChASE/obj/phi.o
# # $(addprefix ../phiChASE/obj/,$(notdir $(MIC_SRCS:.cpp=.o))) 
# # $(wildcard ../phiChASE/obj/*.o)
MIC_DRVS = "../example drivers/phiChASE/main.cpp"

CUDA_SRC_PATH = ../cudaChASE/src
CUDA_INC_PATH = ../cudaChASE/include
CUDA_KER_PATH = ../cudaChASE/kernels
CUDA_OBJ_PATH = ../cudaChASE/obj
CUDA_DRV_PATH = ../example drivers/cudaChASE/
CUDA_SRCS = $(wildcard ../cudaChASE/src/*.cpp)
CUDA_INCS = $(wildcard ../cudaChASE/include/*.h)
CUDA_KERS = $(wildcard ../cudaChASE/kernels/*.cu)
CUDA_OBJS = $(wildcard ../cudaChASE/obj/*.o)
CUDA_DRVS = $(../example drivers/cudaChASE/main.cpp)

ELE_SRC_PATH = ../EleChASE/src
ELE_INC_PATH = ../EleChASE/include
ELE_OBJ_PATH = ../EleChASE/obj
ELE_DRV_PATH = ../example drivers/EleChASE/)
ELE_SRCS = $(wildcard ../EleChASE/src/*.cpp)
ELE_INCS = $(wildcard ../EleChASE/include/*.h)
ELE_OBJS = $(wildcard ../EleChASE/obj/*.o)
ELE_DRVS = $(wildcard ../example drivers/EleChASE/*.cpp)

ifeq ($(ElVersion),EIGHTYFIVE)
	include /lustre/jhome10/slai/slai02/programs/Elemental-0.85/conf/ElVars
	ELEM_COMPILE_FLAGS = $(EL_COMPILE_FLAGS)
	ELEM_LINK_FLAGS = -lkiss_fft -lmetis $(EL_LINK_FLAGS) 
	ELEM_LIBS = $(EL_LIBS)
else 
ifeq ($(ElVersion),EIGHTYFOURDEV)
	include /lustre/jhome10/slai/slai02/programs/elemental-0.84-dev/conf/elemvariables
else	
#	include /lustre/jhome10/slai/slai02/programs/Elemental-0.84-p1/conf/ElemVars
endif
endif

######################### mic

phi-driver: phi ${MIC_OBJS} 
	${MIC_CC} $(MIC_CXX_FLAGS) -o ${EXEC_PATH}/phiChASE ${MIC_OBJS} ${MIC_DRVS} ${MIC_LINKER_FLAGS}

### ${MIC_SRCS} ${MIC_INCS}
phi: 
	${MIC_CC} $(MIC_CXX_FLAGS) -c ${MIC_SRCS} ${MIC_INCS} 
	
#-o ${MIC_OBJS}

clean-phi:
	rm -f ${MIC_OBJS} 
clean-phi-driver: clean phi
	rm -f ${EXEC_PATH}/phiChASE
	
############# cuda	 

cuda-driver: ${CUDA_OBJS} 
	${CUDA_CC} $(CUDA_CXX_FLAGS) ${CUDA_LINKER_FLAGS} -o ${EXEC_PATH}/cuChASE ${CUDA_OBJS} ${CUDA_DRVS} ${MAGMA_LIB}

cuda: ${CUDA_SRCS} ${CUDA_INCS} ${CUDA_KERS}
	${CUDA_CC} ${CUDA_CXX_FLAGS} ${CUDA_INCLUDE} ${MAGMA_INCLUDE} -o ${CUDA_OBJ_PATH}/$@ -c  ${CUDA_SRCS} ${CUDA_INCS}
	${NVCC} ${NVCC_FLAGS}  ${MAGMA_INCLUDE} -o ${CUDA_OBJ_PATH}/$@ -c ${CUDA_KERS}

#cuda: ${CUDA_KERS}
#	${NVCC} ${NVCC_FLAGS}  ${MAGMA_INCLUDE} -o ${CUDA_OBJ_PATH}/$@ -c $<

clean-cuda:
	rm -f ${CUDA_OBJS}
clean-cuda-driver: clean cuda
	rm -f ${EXEC_PATH}/cuChASE

############## elemental

ele: ${ELE_SRCS} ${ELE_INCS}
	${MPICXX} ${ELEM_COMPILE_FLAGS} -D$(ElVersion) -o ${ELE_OBJ_PATH} -c ${ELE_SRCS} ${ELE_INCS}
 
clean-ele:
	rm -f ${ELE_OBJS}
########### existing elechfsi makefile

SRCS=$(wildcard src/*.cpp)
OBJS=$(SRCS:src/%.cpp=obj/.o)


#obj/lanczos.o obj/filter.o obj/chfsi.o

ele-drivers: complex mwi grid_algbs interface


complex: obj/chfsi.o obj/zmain.o
	${MPICXX} ${ELEM_COMPILE_FLAGS} obj/chfsi.o obj/zmain.o -o $(zEXECUTABLE) -D$(ElVersion) ${ELEM_LINK_FLAGS} ${ELEM_LIBS}


mwi: obj/mwi.o
	${MPICXX} ${ELEM_COMPILE_FLAGS} obj/mwi.o -o mwi -D$(ElVersion) ${ELEM_LINK_FLAGS} ${ELEM_LIBS}


interface: obj/chfsi.o obj/el_fl.o obj/intf.o
	${MPICXX} ${ELEM_COMPILE_FLAGS} obj/chfsi.o obj/intf.o obj/el_fl.o -o intf ${ELEM_LINK_FLAGS} ${ELEM_LIBS}


gemm: obj/gemm.o
	${MPICXX} ${ELEM_COMPILE_FLAGS} obj/gemm.o -o gemm ${ELEM_LINK_FLAGS} ${ELEM_LIBS}



grid_algbs: obj/grid_algbs.o
	${MPICXX} ${ELEM_COMPILE_FLAGS} obj/grid_algbs.o -o grid_algbs ${ELEM_LINK_FLAGS} ${ELEM_LIBS}


heig: obj/heig.o
	${MPICXX} ${ELEM_COMPILE_FLAGS} obj/heig.o -o heig ${ELEM_LINK_FLAGS} ${ELEM_LIBS}
	
	
obj/chfsi.o: src/chfsi.cpp
	${MPICXX} ${ELEM_COMPILE_FLAGS} -c $< -D$(ElVersion) -o $@ 

obj/zmain.o: src/zmain.cpp
	${MPICXX} ${ELEM_COMPILE_FLAGS} -c $< -D$(ElVersion) -o $@ 
#	${MPICXX} ${ELEM_COMPILE_FLAGS} -c $< -DELECHFSI_DEBUG -D$(ElVersion) -o $@ 

obj/mwi.o: src/mwi.cpp
	${MPICXX} ${ELEM_COMPILE_FLAGS} -c $< -D$(ElVersion) -o $@ 

obj/el_fl.o: src/elechfsi_fleur.cpp
	${MPICXX} ${ELEM_COMPILE_FLAGS} -c $< -o $@ 

obj/intf.o: src/zmain_interface.cpp
	${MPICXX} ${ELEM_COMPILE_FLAGS} -c $< -o $@ 

obj/gemm.o: src/gemm.cpp
	${MPICXX} ${ELEM_COMPILE_FLAGS} -c $< -o $@ 
	
obj/grid_algbs.o: src/grid_algbs.cpp
	${MPICXX} ${ELEM_COMPILE_FLAGS} -c $< -D$(ElVersion) -o $@ 

obj/heig.o: src/heig.cpp
	${MPICXX} ${ELEM_COMPILE_FLAGS} -c $< -o $@ 

### CLEAN RULES 

objs: phi cuda ele

all: objs phi driver cuda driver ele drivers

clean-objs: clean cuda clean phi clean ele

clean-exec:
	rm -f ${EXEC_PATH}/*

clean: clean obj clean exec

dist-clean: clean
	rm -f $(EXECUTABLE) *~ .depend 

depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	@$(foreach SRC, $(SRCS), $(MPICXX) -MG -MT $(SRC:src/%.cpp=obj/.o) -MM $(SRC) >> .depend;)

#@$(foreach SRC, $(SRCS), $(MPICXX) -MT $(OBJS) -MM $(SRC) >> .depend;)

# include .depend	
