#pragma once

#include "mpi.h"
#ifdef HAS_SCALAPACK
#include "external/scalapackpp/scalapackpp.hpp"
#endif
#ifdef HAS_NCCL
#include <nccl.h>
#include "grid/nccl_utils.hpp"
#endif

/**
 * @page mpi_grid_module chase::grid Namespace
 * This module defines classes and enumerations for creating and managing 
 * a 2D MPI grid, with support for row-major and column-major ordering. 
 * It includes both an abstract base class, `MpiGrid2DBase`, and a templated
 * derived class, `MpiGrid2D`, to configure the grid for use with 
 * MPI communicators.
 *
 * If ScaLAPACK is available, it will create a ScaLAPACK context on top of it.
 * If NCLL is enabled, it will create corresponding NCCL communicators on top of it.
 */

/**
 * @defgroup grid_namespace Grid
 * @{
 * @brief Namespace containing classes and enumerations related to MPI 2D grid setup.
 */

namespace chase {
namespace grid {

/**
 * @ingroup grid_namespace
 * @brief Enumeration to specify the major ordering of the MPI grid.
 *
 * The `GridMajor` enum defines two values to determine the layout of the grid:
 * - `RowMajor`: The grid is row-major.
 * - `ColMajor`: The grid is column-major.
 */
enum class GridMajor {
    RowMajor,
    ColMajor
};

// Abstract base class for MpiGrid2D
/**
 * @ingroup grid_namespace
 * @brief Abstract base class for managing a 2D MPI grid.
 *
 * The `MpiGrid2DBase` class provides an abstract interface for managing
 * an MPI 2D grid. It defines virtual functions for obtaining MPI communicators,
 * grid coordinates, dimensions, and other relevant properties of the grid.
 * It supports optional integration with NCCL and ScaLAPACK.
 */
class MpiGrid2DBase {
public:
    virtual ~MpiGrid2DBase() = default;
    /**
     * @brief Returns the row communicator for the grid.
     * @return MPI_Comm Row communicator.
     */    
    virtual MPI_Comm get_row_comm() const = 0;
    /**
     * @brief Returns the column communicator for the grid.
     * @return MPI_Comm Column communicator.
     */    
    virtual MPI_Comm get_col_comm() const = 0;
    /**
     * @brief Returns the global communicator for the grid.
     * @return MPI_Comm Global communicator.
     */    
    virtual MPI_Comm get_comm() const = 0;
    /**
     * @brief Returns the coordinates of the current process in the grid.
     * @return Pointer to an array containing the coordinates.
     */    
    virtual int* get_coords() = 0;
    /**
     * @brief Returns the dimensions of the grid.
     * @return Pointer to an array containing the grid dimensions.
     */    
    virtual int* get_dims() = 0;
    /**
     * @brief Returns the number of processes in the grid.
     * @return int Number of processes.
     */    
    virtual int get_nprocs() const = 0;
    /**
     * @brief Returns the rank of the current process in the grid.
     * @return int Rank of the current process.
     */    
    virtual int get_myRank() const = 0;
    /**
     * @brief Returns the major ordering of the grid.
     * @return GridMajor The grid's major ordering (RowMajor or ColMajor).
     */    
    virtual GridMajor getGridMajor() const = 0;
#ifdef HAS_NCCL
    /**
     * @brief Returns the NCCL global communicator for the grid.
     * @return MPI_Comm Global communicator.
     */    
    virtual ncclComm_t get_nccl_comm() const = 0;
    /**
     * @brief Returns the NCCL row communicator for the grid.
     * @return MPI_Comm Global communicator.
     */      
    virtual ncclComm_t get_nccl_row_comm() const = 0;
    /**
     * @brief Returns the NCCL column communicator for the grid.
     * @return MPI_Comm Global communicator.
     */       
    virtual ncclComm_t get_nccl_col_comm() const = 0;
#endif
#ifdef HAS_SCALAPACK
    /**
     * @brief Returns the ScaLAPACK context of column communicator.
     * @return MPI_Comm Global communicator.
     */  
    virtual int get_blacs_colcomm_ctxt() = 0;
    /**
     * @brief Returns the ScaLAPACK context of row communicator.
     * @return MPI_Comm Global communicator.
     */      
    virtual int get_blacs_rowcomm_ctxt() = 0;
    /**
     * @brief Returns the ScaLAPACK context of 2D communicator.
     * @return MPI_Comm Global communicator.
     */      
    virtual int get_blacs_comm2D_ctxt() = 0;  
#endif      
};

// Templated MpiGrid2D class
/**
 * @ingroup grid_namespace
 * @brief Templated class for creating and managing a 2D MPI grid.
 *
 * The `MpiGrid2D` class is a templated class derived from `MpiGrid2DBase` that 
 * manages a 2D MPI grid with specified row and column dimensions.
 * The grid can be configured to use either row-major or column-major ordering.
 *
 * @tparam MajorOrder Specifies the major ordering of the grid (RowMajor or ColMajor).
 */
template<GridMajor MajorOrder>
class MpiGrid2D : public MpiGrid2DBase {
public:
    // Constructors and methods...
    /**
     * @brief Constructs a 2D MPI grid with specified row and column dimensions.
     *
     * @param row_dim Number of rows in the grid.
     * @param col_dim Number of columns in the grid.
     * @param comm MPI communicator used to create the grid.
     */    
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
    /**
     * @brief Constructs a 2D MPI grid with dimensions derived from the communicator.
     *
     * @param comm MPI communicator used to create the grid.
     */
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

    /**
     * @brief Destructor to clean up resources associated with the MPI grid.
     */
    ~MpiGrid2D()
    {
#ifdef HAS_NCCL
        CHECK_NCCL_ERROR(ncclCommDestroy(nccl_comm_));
        CHECK_NCCL_ERROR(ncclCommDestroy(nccl_row_comm_));
        CHECK_NCCL_ERROR(ncclCommDestroy(nccl_col_comm_));
#endif        
    } 

    // Implement equality operator
    /**
     * @brief Equality operator to compare two `MpiGrid2D` instances.
     *
     * @param other The other `MpiGrid2D` instance to compare with.
     * @return `true` if the two grids are identical, `false` otherwise.
     */    
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
    /**
     * @brief Inequality operator to compare two `MpiGrid2D` instances.
     *
     * @param other The other `MpiGrid2D` instance to compare with.
     * @return `true` if the two grids are different, `false` otherwise.
     */    
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
    /**
     * @brief Helper function to create a 2D grid with MPI Cartesian topology.
     */
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

    MPI_Comm comm_; ///< MPI communicator for the entire grid
    MPI_Comm row_comm_; ///< MPI communicator for the row dimension
    MPI_Comm col_comm_; ///< MPI communicator for the column dimension
    int dims_[2]={0,0}; ///< Grid dimensions [rows, columns]
    int coords_[2]; ///< Coordinates of the current process in the grid
    int row_procs_; ///< Number of processes in each row
    int col_procs_; ///< Number of processes in each column
    int nprocs_; ///< Total number of processes in the grid
    int myrank_; ///< Rank of the current process within the communicator
#ifdef HAS_NCCL
    ncclComm_t nccl_comm_; ///< NCCL communicator for the grid
    ncclComm_t nccl_row_comm_; ///< NCCL communicator for row dimension
    ncclComm_t nccl_col_comm_; ///< NCCL communicator for column dimension
#endif
#ifdef HAS_SCALAPACK
    int comm2D_ctxt_; ///< ScaLAPACK context for 2D grid
    int colComm1D_ctxt_; ///< ScaLAPACK context for column communicator
    int rowComm1D_ctxt_; ///< ScaLAPACK context for row communicator
#endif
};
/** @} */ // End of grid_namespace

} // namespace grid
} // namespace chase
