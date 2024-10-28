#pragma once

#include "mpi.h"
#ifdef HAS_SCALAPACK
#include "external/scalapackpp/scalapackpp.hpp"
#endif
#ifdef HAS_NCCL
#include <nccl.h>
#include "grid/nccl_utils.hpp"
#endif

namespace chase {
namespace grid {

enum class GridMajor {
    RowMajor,
    ColMajor
};

// Abstract base class for MpiGrid2D
class MpiGrid2DBase {
public:
    virtual ~MpiGrid2DBase() = default;
    virtual MPI_Comm get_row_comm() const = 0;
    virtual MPI_Comm get_col_comm() const = 0;
    virtual MPI_Comm get_comm() const = 0;
    virtual int* get_coords() = 0;
    virtual int* get_dims() = 0;
    virtual int get_nprocs() const = 0;
    virtual int get_myRank() const = 0;
    virtual GridMajor getGridMajor() const = 0;
#ifdef HAS_NCCL
    virtual ncclComm_t get_nccl_comm() const = 0;
    virtual ncclComm_t get_nccl_row_comm() const = 0;
    virtual ncclComm_t get_nccl_col_comm() const = 0;
#endif
#ifdef HAS_SCALAPACK
    virtual int get_blacs_colcomm_ctxt() = 0;
    virtual int get_blacs_rowcomm_ctxt() = 0;
    virtual int get_blacs_comm2D_ctxt() = 0;  
#endif      
};

// Templated MpiGrid2D class
template<GridMajor MajorOrder>
class MpiGrid2D : public MpiGrid2DBase {
public:
    // Constructors and methods...
    MpiGrid2D(int row_dim, int col_dim, MPI_Comm comm)
        : comm_(comm) {
        dims_[0] = row_dim;
        dims_[1] = col_dim;
        MPI_Comm_size(comm_, &nprocs_);
        MPI_Comm_rank(comm_, &myrank_);
#ifdef HAS_NCCL
        MPI_Comm shm_comm_;
        MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm_);
        int shm_rank_;
        MPI_Comm_rank(shm_comm_, &shm_rank_);  
        int num_devices = -1;
        cudaGetDeviceCount(&num_devices);
        //std::cout << "visible cuda devices = " << num_devices << std::endl;
        cudaSetDevice(shm_rank_);
#endif
        create2DGrid();
    }

    MpiGrid2D(MPI_Comm comm)
        : comm_(comm) {
        MPI_Comm_size(comm_, &nprocs_);
        MPI_Dims_create(nprocs_, 2, dims_);
        MPI_Comm_rank(comm_, &myrank_);

#ifdef HAS_NCCL
        MPI_Comm shm_comm_;
        MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm_);
        int shm_rank_;
        MPI_Comm_rank(shm_comm_, &shm_rank_);  
        int num_devices = -1;
        cudaGetDeviceCount(&num_devices);
        //std::cout << "visible cuda devices = " << num_devices << std::endl;
        cudaSetDevice(shm_rank_);
#endif         
        create2DGrid();
  
    }

    ~MpiGrid2D()
    {
#ifdef HAS_NCCL
        CHECK_NCCL_ERROR(ncclCommDestroy(nccl_comm_));
        CHECK_NCCL_ERROR(ncclCommDestroy(nccl_row_comm_));
        CHECK_NCCL_ERROR(ncclCommDestroy(nccl_col_comm_));
#endif        
    } 

    // Implement equality operator
    bool operator==(const MpiGrid2D<MajorOrder>& other) const {
        // Compare communicators
        int result;
        MPI_Comm_compare(comm_, other.comm_, &result);
        if (result != MPI_IDENT) return false;

        // Compare row and column communicators
        MPI_Comm_compare(row_comm_, other.row_comm_, &result);
        if (result != MPI_IDENT) return false;
        MPI_Comm_compare(col_comm_, other.col_comm_, &result);
        if (result != MPI_IDENT) return false;

        // Compare grid dimensions and coordinates
        return (dims_[0] == other.dims_[0] &&
                dims_[1] == other.dims_[1] &&
                coords_[0] == other.coords_[0] &&
                coords_[1] == other.coords_[1]);
    }

    // Implement inequality operator
    bool operator!=(const MpiGrid2D<MajorOrder>& other) const {
        return !(*this == other);
    }
    
    MPI_Comm get_row_comm() const override { return row_comm_; }
    MPI_Comm get_col_comm() const override { return col_comm_; }
    MPI_Comm get_comm() const override { return comm_; }
    int get_nprocs() const override { return nprocs_; }
    int get_myRank() const override { return myrank_; } 

    int* get_coords() override { return coords_; }
    int* get_dims() override { return dims_; }
    GridMajor getGridMajor() const override { return MajorOrder; }

#ifdef HAS_NCCL
    ncclComm_t get_nccl_comm() const override { return nccl_comm_; };
    ncclComm_t get_nccl_row_comm() const override { return nccl_row_comm_; };
    ncclComm_t get_nccl_col_comm() const override { return nccl_col_comm_; };
#endif

#ifdef HAS_SCALAPACK
    int get_blacs_colcomm_ctxt() override { return colComm1D_ctxt_; }
    int get_blacs_rowcomm_ctxt() override { return rowComm1D_ctxt_; }
    int get_blacs_comm2D_ctxt() override { return comm2D_ctxt_; }
#endif

private:
    void create2DGrid() {
        int tmp_dims_[2];
        int tmp_coords_[2];
        tmp_dims_[0] = dims_[0];
        tmp_dims_[1] = dims_[1];

        if (MajorOrder == GridMajor::ColMajor) {
            std::swap(tmp_dims_[0], tmp_dims_[1]);
        }

        int periodic[] = {0, 0};
        int reorder = 0;
        int free_coords[2];

        MPI_Comm cartComm;
        MPI_Cart_create(comm_, 2, tmp_dims_, periodic, reorder, &cartComm);

        int rank_, nprocs_;
        MPI_Comm_size(cartComm, &nprocs_);
        MPI_Comm_rank(cartComm, &rank_);
        MPI_Cart_coords(cartComm, rank_, 2, tmp_coords_);

        if (MajorOrder == GridMajor::ColMajor) {
            coords_[1] = tmp_coords_[0];
            coords_[0] = tmp_coords_[1];
        } else {
            coords_[1] = tmp_coords_[1];
            coords_[0] = tmp_coords_[0];
        }

        free_coords[0] = (MajorOrder == GridMajor::ColMajor) ? 1 : 0;
        free_coords[1] = (MajorOrder == GridMajor::ColMajor) ? 0 : 1;
        MPI_Cart_sub(cartComm, free_coords, &row_comm_);
        MPI_Comm_size(row_comm_, &row_procs_);

        free_coords[0] = (MajorOrder == GridMajor::ColMajor) ? 0 : 1;
        free_coords[1] = (MajorOrder == GridMajor::ColMajor) ? 1 : 0;
        MPI_Cart_sub(cartComm, free_coords, &col_comm_);
        MPI_Comm_size(col_comm_, &col_procs_);

        MPI_Comm_free(&cartComm);
#ifdef HAS_NCCL
        ncclUniqueId id;
        if (myrank_ == 0) {
            CHECK_NCCL_ERROR(ncclGetUniqueId(&id));
        }
        MPI_Bcast((void *) &id, sizeof(id), MPI_BYTE, 0, comm_);
        CHECK_NCCL_ERROR(ncclCommInitRank(&nccl_comm_, nprocs_, id, myrank_));

        ncclUniqueId nccl_id, nccl_ids[nprocs_];
        CHECK_NCCL_ERROR(ncclGetUniqueId(&nccl_id));
        MPI_Allgather(&nccl_id, sizeof(ncclUniqueId), MPI_UINT8_T, &nccl_ids[0],
                      sizeof(ncclUniqueId), MPI_UINT8_T, comm_);
        //nccl row comm
        for (auto i = 0; i < dims_[0]; i++)
        {
            if (coords_[0] == i)
            {
                CHECK_NCCL_ERROR(ncclCommInitRank(&nccl_row_comm_, dims_[1], nccl_ids[i],
                                 coords_[1]));
            }
        }

        //nccl column comm
        ncclUniqueId nccl_id_2, nccl_ids_2[nprocs_];
        CHECK_NCCL_ERROR(ncclGetUniqueId(&nccl_id_2));
        MPI_Allgather(&nccl_id_2, sizeof(ncclUniqueId), MPI_UINT8_T,
                      &nccl_ids_2[0], sizeof(ncclUniqueId), MPI_UINT8_T, comm_);

        for (auto i = 0; i < dims_[1]; i++)
        {
            if (coords_[1] == i)
            {
                CHECK_NCCL_ERROR(ncclCommInitRank(&nccl_col_comm_, dims_[0],
                                 nccl_ids_2[i * dims_[0]], coords_[0]));
            }
        }

#endif

#ifdef HAS_SCALAPACK
        int zero = 0;
        int one = 1;
        int ictxt;
        chase::linalg::scalapackpp::blacs_get_(&zero, &zero, &ictxt);
        colComm1D_ctxt_ = ictxt;
        int userMap[dims_[0]];
        if(MajorOrder == GridMajor::ColMajor)
        {
            for (int i = 0; i < dims_[0]; i++)
            {
                userMap[i] = (myrank_ / dims_[0]) * dims_[0] + i;
            }

        }else
        {
            for (int i = 0; i < dims_[0]; i++)
            {
                userMap[i] = (myrank_ % dims_[1]) + dims_[1] * i;
            }
        }

        chase::linalg::scalapackpp::blacs_gridmap_(&colComm1D_ctxt_, userMap, &dims_[0], &dims_[0], &one);

        int ictxt_2;
        chase::linalg::scalapackpp::blacs_get_(&zero, &zero, &ictxt_2);
        comm2D_ctxt_ = ictxt_2;
        if (MajorOrder == GridMajor::RowMajor){
            char major = 'R';
            chase::linalg::scalapackpp::blacs_gridinit_(&comm2D_ctxt_, &major, &dims_[0], &dims_[1]);
        }else
        {
            char major = 'C';
            chase::linalg::scalapackpp::blacs_gridinit_(&comm2D_ctxt_, &major, &dims_[0], &dims_[1]);          
        }
#endif

    }

    MPI_Comm comm_;
    MPI_Comm row_comm_;
    MPI_Comm col_comm_;
    int dims_[2]={0,0};
    int coords_[2];
    int row_procs_;
    int col_procs_;
    int nprocs_;
    int myrank_;
#ifdef HAS_NCCL
    ncclComm_t nccl_comm_;
    ncclComm_t nccl_row_comm_;
    ncclComm_t nccl_col_comm_;
#endif
#ifdef HAS_SCALAPACK
    int comm2D_ctxt_;
    int colComm1D_ctxt_;
    int rowComm1D_ctxt_; //impl to be added later
#endif

};

} // namespace grid
} // namespace chase
