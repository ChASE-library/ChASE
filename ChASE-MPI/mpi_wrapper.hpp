#pragma once
#include <map>
#if defined(HAS_NCCL)
#include <nccl.h>
#endif
#include <mpi.h>

namespace chase
{
namespace mpi
{
#define MPI_BACKEND 0
#if defined(HAS_NCCL)
#define NCCL_BACKEND 1
#endif

#define CPY_H 0
#define CPY_D 1

typedef MPI_Op Op_1;
typedef MPI_Op Op_1;
typedef MPI_Datatype datatype_1;
typedef MPI_Comm comm_1;
#if defined(HAS_NCCL)
typedef ncclRedOp_t Op_2;
typedef ncclDataType_t datatype_2;
typedef ncclComm_t comm_2;
#else
typedef MPI_Op Op_2;
typedef MPI_Datatype datatype_2;
typedef MPI_Comm comm_2;
#endif


class Comm{
        public:
                Comm(){
#if defined(HAS_NCCL)
                    opt_map_[MPI_MAX] = ncclMax;
                    opt_map_[MPI_MIN] = ncclMin;
                    opt_map_[MPI_SUM] = ncclSum;
                    opt_map_[MPI_PROD] = ncclProd;
                    datatype_map_[MPI_DOUBLE] = ncclDouble;
                    datatype_map_[MPI_FLOAT] = ncclFloat;
                    datatype_map_[MPI_COMPLEX] = ncclFloat;
                    datatype_map_[MPI_DOUBLE_COMPLEX] = ncclDouble;
#else
                    opt_map_[MPI_MAX] = MPI_MAX;
                    opt_map_[MPI_MIN] = MPI_MIN;
                    opt_map_[MPI_SUM] = MPI_SUM;
                    opt_map_[MPI_PROD] = MPI_PROD;
                    datatype_map_[MPI_DOUBLE] = MPI_DOUBLE;
                    datatype_map_[MPI_FLOAT] = MPI_FLOAT;
                    datatype_map_[MPI_COMPLEX] = MPI_COMPLEX;
                    datatype_map_[MPI_DOUBLE_COMPLEX] = MPI_DOUBLE_COMPLEX;
#endif
                }

                ~Comm(){}
                void add(comm_1 mpi_comm, comm_2 nccl_comm)
                {
                   comm_map_[mpi_comm] = nccl_comm;
                }

                comm_2 get_comm(comm_1 key){
                    return comm_map_.at(key);
                }

                Op_2 get_Op(MPI_Op op){
                    return opt_map_.at(op);
                }

                datatype_2 get_datatype(MPI_Datatype type){
                    return datatype_map_.at(type);
                }
        private:
                std::map<comm_1, comm_2> comm_map_;
                std::map<Op_1, Op_2> opt_map_;
                std::map<datatype_1, datatype_2> datatype_map_;

};

typedef Comm Comm_t;

template<typename T>
void AllReduce(int backend, T *send_data, T *recv_data, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, Comm_t env){
    switch(backend)
    {
#if defined(HAS_NCCL)
        case NCCL_BACKEND:
            ncclAllReduce(send_data, recv_data, int(sizeof(T)/sizeof(Base<T>)) * count, env.get_datatype(datatype), env.get_Op(op), env.get_comm(comm), NULL);
            break;
#endif
        case MPI_BACKEND:
            MPI_Allreduce(send_data, recv_data, count, datatype, op, comm);
            break;
    }
}

template<typename T>
void AllReduce(int backend, T *data, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, Comm_t env){
    switch(backend)
    {
#if defined(HAS_NCCL)
        case NCCL_BACKEND:
            ncclAllReduce(data, data, int(sizeof(T)/sizeof(Base<T>)) * count, env.get_datatype(datatype), env.get_Op(op), env.get_comm(comm), NULL);
            break;
#endif
        case MPI_BACKEND:
            MPI_Allreduce(MPI_IN_PLACE, data, count, datatype, op, comm);
            break;
    }
}

void Memcpy(int mode, void* dst, const void* src, std::size_t count)
{
    switch(mode)
    {
	case CPY_H:
	    std::memcpy(dst, src, count);
	    break;
#if defined(CUDA_AWARE)
        case CPY_D:
            cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
            break;
#endif	    
    }	    
}


} // namespace mpi
} // namespace chase
