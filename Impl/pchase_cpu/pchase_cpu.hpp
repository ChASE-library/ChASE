// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstring>
#include <memory>
#include <random>
#include <vector>
#include "algorithm/chaseBase.hpp"
#include "linalg/matrix/matrix.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/internal/mpi/mpi_kernels.hpp"
#ifdef HAS_SCALAPACK
#include "external/scalapackpp/scalapackpp.hpp"
#endif
#include "algorithm/types.hpp"
#include "../../linalg/internal/typeTraits.hpp"

using namespace chase::linalg;

namespace chase
{
namespace Impl
{
/**
 * @page pChASECPU
 * 
 * @section intro_sec Introduction
 * This class implements the CPU-based parallel version of the Chase algorithm using MPI. 
 * It inherits from `ChaseBase` and provides methods for solving generalized eigenvalue problems 
 * with MPI-based parallelism. The class operates on matrix and multi-vector data types, leveraging MPI 
 * for communication across different processes in a distributed computing environment.
 * 
 * @section constructor_sec Constructors and Destructor
 * The constructor and destructor for the `pChASECPU` class manage the initialization of the matrix 
 * data, multi-vectors, and MPI communication. The constructor also ensures that the matrix is square 
 * and that the matrix and eigenvectors are mapped to the same MPI grid.
 * 
 * @section members_sec Private Members
 * Private members include matrix data, multi-vectors, configuration settings, and MPI-specific 
 * information such as rank, size, coordinates, and dimensions. These members are initialized 
 * during construction.
 */

/**
 * @brief CPU-based parallel Chase algorithm with MPI.
 * 
 * This class solves generalized eigenvalue problems using a parallel implementation of the Chase 
 * algorithm. It leverages MPI for parallel processing and works with matrix and multi-vector data 
 * types.
 * 
 * @tparam MatrixType The matrix type, such as `chase::distMatrix::RedundantMatrix`.
 * @tparam InputMultiVectorType The input multi-vector type, typically used for eigenvectors.
 */
template <typename MatrixType, typename InputMultiVectorType, typename BackendType = chase::grid::backend::MPI>
class pChASECPU : public ChaseBase<typename MatrixType::value_type>
{
    using T = typename MatrixType::value_type;
    using ResultMultiVectorType = typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;
    static_assert(std::is_same_v<BackendType, chase::grid::backend::MPI>,
                  "BackendType must be chase::grid::backend::MPI");
public:
    /**
     * @brief Constructor for the pChASECPU class.
     * 
     * Initializes the Chase algorithm with the given matrix, eigenvector, and Ritz values. The matrix 
     * and eigenvectors must be mapped to the same MPI grid. The constructor also verifies that the 
     * matrix is square and sets up the necessary data structures for the algorithm.
     * 
     * @param nev The number of eigenvalues to compute.
     * @param nex The number of additional eigenvalues.
     * @param H Pointer to the matrix to solve.
     * @param V Pointer to the input multi-vector of eigenvectors.
     * @param ritzv Pointer to the Ritz values used in the algorithm.
     */
    pChASECPU(std::size_t nev,
                std::size_t nex,
                MatrixType *H,
                InputMultiVectorType *V,
                chase::Base<T> *ritzv
                ): nev_(nev), nex_(nex), nevex_(nev + nex), config_(H->g_rows(), nev, nex), N_(H->g_rows())
    { 
        if(H->g_rows() != H->g_cols())
        {
            throw std::runtime_error("ChASE requires the matrix solved to be squared");
        }

        if( H->getMpiGrid() != V->getMpiGrid())
        {   
            throw std::runtime_error("ChASE requires the matrix and eigenvectors mapped to same MPI grid");
        }

        Hmat_ = H;
        V1_ = V;
        
        //V2_ = new InputMultiVectorType(Hmat_->g_rows(), nevex_, Hmat_->getMpiGrid_shared_ptr());
        V2_ = V1_->template clone2<InputMultiVectorType>();
        W1_ = V1_->template clone2<ResultMultiVectorType>();
        W2_ = V1_->template clone2<ResultMultiVectorType>();

        ritzv_ = std::make_unique<chase::distMatrix::RedundantMatrix<chase::Base<T>>>(nevex_, 1, nevex_, ritzv, Hmat_->getMpiGrid_shared_ptr());
        resid_ = std::make_unique<chase::distMatrix::RedundantMatrix<chase::Base<T>>>(nevex_, 1, Hmat_->getMpiGrid_shared_ptr());

        MPI_Comm_rank(Hmat_->getMpiGrid()->get_comm(), &my_rank_);
        MPI_Comm_size(Hmat_->getMpiGrid()->get_comm(), &nprocs_);
        coords_ = Hmat_->getMpiGrid()->get_coords();
        dims_ = Hmat_->getMpiGrid()->get_dims();

        if constexpr (std::is_same<typename MatrixType::hermitian_type, chase::matrix::QuasiHermitian>::value)
         {
                is_sym_ = false;
                is_pseudoHerm_ = true;
                //Quasi Hermitian matrices require more space for the dual basis
        	A_ = std::make_unique<chase::distMatrix::RedundantMatrix<T>>(nevex_, 3*nevex_, Hmat_->getMpiGrid_shared_ptr());
         }
         else
         {
                is_sym_ = true;
        	is_pseudoHerm_ = false;
        	A_ = std::make_unique<chase::distMatrix::RedundantMatrix<T>>(nevex_, nevex_, Hmat_->getMpiGrid_shared_ptr());
         }
    }
    /**
     * @brief Deleted copy constructor.
     * 
     * The copy constructor is deleted to prevent copying of this class, as it manages
     * unique resources such as MPI communication.
     */
    pChASECPU(const pChASECPU&) = delete;
    /**
     * @brief Destructor for the pChASECPU class.
     * 
     * Cleans up any resources allocated by the constructor, including matrix data and MPI-specific 
     * information.
     */
    ~pChASECPU() {}

    std::size_t GetN() const override { return N_; }

    std::size_t GetNev() override { return nev_; }
    
    std::size_t GetNex() override { return nex_; }

    chase::Base<T>* GetRitzv() override { return ritzv_->l_data(); }
    chase::Base<T>* GetResid() override { return resid_->l_data(); }
    ChaseConfig<T>& GetConfig() override { return config_; }
    int get_nprocs() override { return nprocs_; }
    int get_rank() { return my_rank_; }

    void loadProblemFromFile(std::string filename)
    {
        Hmat_->readFromBinaryFile(filename);
    }

    void saveProblemToFile(std::string filename)
    {
        Hmat_->saveToBinaryFile(filename);
    }

#ifdef CHASE_OUTPUT
    //! Print some intermediate infos during the solving procedure
    void Output(std::string str) override
    {
        if(my_rank_ == 0)
        {
            std::cout << str;
        }
    }
#endif
    bool checkSymmetryEasy() override
    {
        is_sym_ = chase::linalg::internal::cpu_mpi::checkSymmetryEasy(*Hmat_);  
        return is_sym_;
    }

    bool isSym() override {return is_sym_;} 
    
    bool checkPseudoHermicityEasy() override
    {
        return is_pseudoHerm_;
    }
    
    bool isPseudoHerm() override { return is_pseudoHerm_; }

    void symOrHermMatrix(char uplo) override
    {
        chase::linalg::internal::cpu_mpi::symOrHermMatrix(uplo, *Hmat_);   
    }

    void Start() override
    {
        locked_ = 0;
    }

    void initVecs(bool random) override
    {
        if (random)
        {
            std::mt19937 gen(1337.0 + coords_[0]);
            std::normal_distribution<> d;

            for (auto j = 0; j < V1_->l_ld() * V1_->l_cols(); j++)
            {
                auto rnd = getRandomT<T>([&]() { return d(gen); });
                V1_->l_data()[j] = rnd;
            }
        }
        
        chase::linalg::lapackpp::t_lacpy('A', 
                                         V1_->l_rows(), 
                                         V1_->l_cols(), 
                                         V1_->l_data(), 
                                         V1_->l_ld(), 
                                         V2_->l_data(), 
                                         V2_->l_ld());
        next_ = NextOp::bAc;
	
    }

    void Lanczos(std::size_t m, chase::Base<T>* upperb) override 
    {
        chase::linalg::internal::cpu_mpi::lanczos_dispatch(m, 
                                              *Hmat_, 
                                              *V1_, 
                                              upperb);
    }

    void Lanczos(std::size_t M, std::size_t numvec, chase::Base<T>* upperb,
                         chase::Base<T>* ritzv, chase::Base<T>* Tau, chase::Base<T>* ritzV) override
    {
        chase::linalg::internal::cpu_mpi::lanczos_dispatch(M, 
              	                              numvec, 
                       	                      *Hmat_, 
                               	              *V1_, 
                                       	      upperb, 
                                      	      ritzv, 
                                      	      Tau, 
                                       	      ritzV);
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        T One = T(1.0);
        T Zero = T(0.0);
        chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                      CblasNoTrans, 
                                      CblasNoTrans, 
                                      V1_->l_rows(), 
                                      idx, 
                                      m, 
                                      &One,
                                      V1_->l_data(), 
                                      V1_->l_ld(), 
                                      ritzVc, 
                                      m, 
                                      &Zero, 
                                      V2_->l_data(), 
                                      V2_->l_ld());

        chase::linalg::lapackpp::t_lacpy('A',
                                         V2_->l_rows(),
                                         m,
                                         V2_->l_data(),
                                         V2_->l_ld(),
                                         V1_->l_data(),
                                         V1_->l_ld());
    }

    void Shift(T c, bool isunshift = false) override 
    {
        if(isunshift)
        {
            next_ = NextOp::bAc;
        }
        chase::linalg::internal::cpu_mpi::shiftDiagonal(*Hmat_, c);

#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value)
        {
            auto min = *std::min_element(resid_->l_data() + locked_, resid_->l_data() + nev_);
            bool shouldEnableSP = (min > 1e-3 && !isunshift);
            auto updatePrecision = [&](auto& mat, bool copyback = false) {
                if (shouldEnableSP) {
                    mat->enableSinglePrecision();
                } else if (mat->isSinglePrecisionEnabled()) {
                    mat->disableSinglePrecision(copyback);
                }
            };

            // Update precision for all matrices
            updatePrecision(Hmat_);
            updatePrecision(V1_, true);  // Special case for V1_
            updatePrecision(W1_);

            // Message on enabling single precision
            if (shouldEnableSP && my_rank_ == 0 && !isunshift) {
                std::cout << "Enable Single Precision in Filter" << std::endl;
            }
            
        }
#endif
        
    }

    void HEMM(std::size_t block, T alpha, T beta, std::size_t offset) override 
    {

#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value)
        {
            using singlePrecisionT = typename chase::ToSinglePrecisionTrait<T>::Type;
            auto min = *std::min_element(resid_->l_data() + locked_, resid_->l_data() + nev_);
            
            if(min > 1e-3)
            {
                auto Hmat_sp = Hmat_->getSinglePrecisionMatrix();
                auto V1_sp = V1_->getSinglePrecisionMatrix();
                auto W1_sp = W1_->getSinglePrecisionMatrix();
                singlePrecisionT alpha_sp = static_cast<singlePrecisionT>(alpha);
                singlePrecisionT beta_sp = static_cast<singlePrecisionT>(beta);  
                //static_assert(std::is_same_v<std::remove_pointer_t<decltype(V1_sp)>, chase::distMultiVector::DistMultiVectorBlockCyclic1D<std::complex<float>, chase::distMultiVector::CommunicatorType::column, chase::platform::CPU>>, "Type mismatch in singlePrecisionT");

                if (next_ == NextOp::bAc)
                {
                    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha_sp, 
                                                                                *Hmat_sp, 
                                                                                *V1_sp, 
                                                                                &beta_sp, 
                                                                                *W1_sp, 
                                                                                offset + locked_, 
                                                                                block);
                    next_ = NextOp::cAb;
                }
                else
                {
                    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors<singlePrecisionT>(&alpha_sp, 
                                                                                *Hmat_sp, 
                                                                                *W1_sp, 
                                                                                &beta_sp, 
                                                                                *V1_sp, 
                                                                                offset + locked_, 
                                                                                block);          
                    next_ = NextOp::bAc;

                }                              
            }
            else
            {
                if (next_ == NextOp::bAc)
                {
                    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, 
                                                                                *Hmat_, 
                                                                                *V1_, 
                                                                                &beta, 
                                                                                *W1_, 
                                                                                offset + locked_, 
                                                                                block);
                    next_ = NextOp::cAb;
                }
                else
                {
                    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, 
                                                                                *Hmat_, 
                                                                                *W1_, 
                                                                                &beta, 
                                                                                *V1_, 
                                                                                offset + locked_, 
                                                                                block);            
                    next_ = NextOp::bAc;

                }                
            }
        }        
        else
#endif
        {
            if (next_ == NextOp::bAc)
            {
                chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, 
                                                                            *Hmat_, 
                                                                            *V1_, 
                                                                            &beta, 
                                                                            *W1_, 
                                                                            offset + locked_, 
                                                                            block);
                next_ = NextOp::cAb;
            }
            else
            {
                chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, 
                                                                            *Hmat_, 
                                                                            *W1_, 
                                                                            &beta, 
                                                                            *V1_, 
                                                                            offset + locked_, 
                                                                            block);            
                next_ = NextOp::bAc;

            }                
                    
        }    
    }

    void QR(std::size_t fixednev, chase::Base<T> cond) override 
    {

        int disable = config_.DoCholQR() ? 0 : 1;
        char* cholddisable = getenv("CHASE_DISABLE_CHOLQR");
        if (cholddisable) {
            disable = std::atoi(cholddisable);
        }

        Base<T> cond_threshold_upper = (sizeof(Base<T>) == 8) ? 1e8 : 1e4;
        Base<T> cond_threshold_lower = (sizeof(Base<T>) == 8) ? 2e1 : 1e1;

        char* chol_threshold = getenv("CHASE_CHOLQR1_THLD");
        if (chol_threshold)
        {
            cond_threshold_lower = std::atof(chol_threshold);
        }

        //int display_bounds = 0;
        //char* display_bounds_env = getenv("CHASE_DISPLAY_BOUNDS");
        //if (display_bounds_env)
        //{
        //    display_bounds = std::atoi(display_bounds_env);
        //}

	
        if constexpr (std::is_same<typename MatrixType::hermitian_type, chase::matrix::QuasiHermitian>::value)
        {
                /* The right eigenvectors are not orthonormal in the QH case, but S-orthonormal.
                 * Therefore, we S-orthonormalize the locked vectors against the current subspace
                 * By flipping the sign of the lower part of the locked vectors. */
                chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(*V1_, 0, locked_);
                /* We do not need to flip back the sign of the locked vectors since they are stored 
                 * in Vec2_ and will replace the fliped ones of Vec1_ at the end of QR. */
        }

#ifdef ChASE_DISPLAY_COND_V_SVD
        if constexpr (std::is_same<typename MatrixType::matrix_type, chase::distMatrix::BlockCyclic>::value &&
                      std::is_same<typename MatrixType::hermitian_type, chase::matrix::Hermitian>::value)
        {
            auto V_tmp = V1_->template clone2<InputMultiVectorType>(V1_->g_rows(), V1_->g_cols() - locked_);
            std::memcpy(V_tmp->l_data(), V1_->l_data() + locked_ * V1_->l_ld(), (V1_->g_cols() - locked_) * V1_->l_ld() * sizeof(T));
            auto cond_v = chase::linalg::internal::cpu_mpi::computeConditionNumber(*V_tmp);
            if(my_rank_ == 0){
                std::cout << "Exact condition number of V from SVD: " << cond_v << std::endl;
            }
        }
#endif

        if (disable == 1)
        {
#ifdef HAS_SCALAPACK
            chase::linalg::internal::cpu_mpi::houseHoulderQR(*V1_);
#else
            throw std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
#endif
        }
        else
        {
#ifdef CHASE_OUTPUT
        if(my_rank_ == 0){
            std::cout << std::setprecision(2) << "cond(V): " << cond << std::endl;
        }
#endif
            //if (display_bounds != 0)
            //{
            //  dla_->estimated_cond_evaluator(locked_, cond);
            //}
            int info = 1;

            if (cond > cond_threshold_upper)
            {
                info = chase::linalg::internal::cpu_mpi::shiftedcholQR2(V1_->g_rows(),
                                                                    V1_->l_rows(), 
                                                                    V1_->l_cols(), 
                                                                    V1_->l_data(),  
                                                                    V1_->l_ld(), 
                                                                    V1_->getMpiGrid()->get_col_comm(), 
                                                                    A_->l_data());
            }
            else if(cond < cond_threshold_lower)
            {
                info = chase::linalg::internal::cpu_mpi::cholQR1(V1_->l_rows(), 
                                                             V1_->l_cols(), 
                                                             V1_->l_data(),  
                                                             V1_->l_ld(), 
                                                             V1_->getMpiGrid()->get_col_comm());
            }
            else
            {
                info = chase::linalg::internal::cpu_mpi::cholQR2(V1_->l_rows(), 
                                                             V1_->l_cols(), 
                                                             V1_->l_data(),  
                                                             V1_->l_ld(), 
                                                             V1_->getMpiGrid()->get_col_comm(),
                                                             A_->l_data()); 
            }

            if (info != 0)
            {
#ifdef HAS_SCALAPACK
#ifdef CHASE_OUTPUT
                if(my_rank_ == 0){
                    std::cout << "CholeskyQR doesn't work, Househoulder QR will be used." << std::endl;
                }
#endif
                chase::linalg::internal::cpu_mpi::houseHoulderQR(*V1_);
#else
                throw std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
#endif
            }
        }

        chase::linalg::lapackpp::t_lacpy('A',
                                         V2_->l_rows(),
                                         locked_,
                                         V2_->l_data(),
                                         V2_->l_ld(),
                                         V1_->l_data(),
                                         V1_->l_ld());

        chase::linalg::lapackpp::t_lacpy('A',
                                         V2_->l_rows(),
                                         nevex_ - locked_,
                                         V1_->l_data() + V1_->l_ld() * locked_,
                                         V1_->l_ld(),
                                         V2_->l_data() + V2_->l_ld() * locked_,
                                         V2_->l_ld());                                              
	
    }

    void RR(chase::Base<T>* ritzv, std::size_t block) override 
    {
        chase::linalg::internal::cpu_mpi::rayleighRitz_dispatch(*Hmat_, 
                                                   *V1_, 
                                                   *V2_, 
                                                   *W1_, 
                                                   *W2_, 
                                                   ritzv_->l_data(), 
                                                   locked_, 
                                                   block,
                                                   A_.get());

        chase::linalg::lapackpp::t_lacpy('A',
                                         V2_->l_rows(),
                                         block,
                                         V1_->l_data() + locked_ * V1_->l_ld(),
                                         V1_->l_ld(),
                                         V2_->l_data() + locked_ * V2_->l_ld(),
                                         V2_->l_ld());   
    }
    
    void Sort(chase::Base<T> * ritzv, chase::Base<T> * residLast, chase::Base<T> * resid) override
    {

    }

    void Resd(chase::Base<T>* ritzv, chase::Base<T>* resd, std::size_t fixednev) override 
    {
        chase::linalg::internal::cpu_mpi::residuals(*Hmat_,
                                                *V1_,
                                                *V2_,
                                                *W1_,
                                                *W2_,
                                                ritzv_->l_data(),
                                                resid_->l_data(),
                                                locked_,
                                                nevex_ - locked_);   
    }

    void Swap(std::size_t i, std::size_t j) override 
    {
        V1_->swap_ij(i, j);
        V2_->swap_ij(i, j);
    }

    void Lock(std::size_t new_converged) override 
    {
        locked_ += new_converged;
    }

    void End() override { }

private:
    /**
    * @brief Enum to represent the next operation to be performed.
    * 
    * This enum specifies the two possible operations in the Chase algorithm: `cAb` and `bAc`.
    */
    enum NextOp
    {
        cAb, /**< Operation cAb: specific matrix-vector multiplication. */
        bAc  /**< Operation bAc: another specific matrix-vector multiplication. */
    };

    /**
    * @brief The current operation to be performed.
    * 
    * This member stores the next operation to be executed based on the algorithm's current step.
    * It can be set to either `cAb` or `bAc`, depending on the context of the computation.
    */
    NextOp next_; 

    /**
    * @brief Flag indicating if the matrix is symmetric.
    * 
    * This boolean value is used to track whether the matrix being processed is symmetric.
    * It influences certain algorithmic steps to optimize performance or correctness.
    */
    bool is_sym_; 
    /**
    * @brief Flag indicating if the matrix is pseudo-hermitian.
    * 
    * This boolean value is used to track whether the matrix being processed is pseudo-hermitian.
    * It influences certain algorithmic steps to optimize performance or correctness.
    */
    bool is_pseudoHerm_; 

    /**
    * @brief The number of eigenvalues to compute.
    * 
    * This member holds the number of eigenvalues (nev_) that the algorithm will compute.
    */
    std::size_t nev_; 

    /**
    * @brief The number of additional eigenvalues.
    * 
    * This member holds the number of extra eigenvalues (nex_) to compute in addition to nev_.
    */
    std::size_t nex_; 

    /**
    * @brief The total number of eigenvalues.
    * 
    * This member holds the total number of eigenvalues to compute, which is the sum of `nev_` and `nex_`.
    */
    std::size_t nevex_; 

    /**
    * @brief Lock state for the algorithm.
    * 
    * This value indicates whether certain parameters are locked, preventing changes during the 
    * algorithm's execution.
    */
    std::size_t locked_; 

    /**
    * @brief The number of rows in the matrix.
    * 
    * This member holds the number of rows in the matrix H (also referred to as `N_`).
    */
    std::size_t N_; 

    /**
    * @brief The total number of MPI processes.
    * 
    * This integer stores the total number of processes involved in the parallel computation.
    */
    int nprocs_; 

    /**
    * @brief The rank of the current MPI process.
    * 
    * This integer stores the rank of the current process within the MPI communicator.
    */
    int my_rank_; 

    /**
    * @brief The MPI process coordinates.
    * 
    * This pointer stores the coordinates of the current process in the MPI grid.
    */
    int *coords_; 

    /**
    * @brief The dimensions of the MPI grid.
    * 
    * This pointer stores the dimensions of the MPI grid, used for distributing data across processes.
    */
    int *dims_; 

    /**
    * @brief Pointer to the matrix H.
    * 
    * This member points to the matrix `H` that is being solved in the algorithm.
    */
    MatrixType *Hmat_; 

    /**
    * @brief Pointer to the input multi-vector of eigenvectors.
    * 
    * This pointer stores the input eigenvectors used for the Chase algorithm.
    */
    InputMultiVectorType *V1_; 

    /**
    * @brief Unique pointer to a cloned input multi-vector.
    * 
    * This unique pointer holds a copy of the input multi-vector `V1_`, used for intermediate computations.
    */
    std::unique_ptr<InputMultiVectorType> V2_; 

    /**
    * @brief Unique pointer to the first result multi-vector.
    * 
    * This unique pointer holds a result multi-vector that is used for storing intermediate computation results.
    */
    std::unique_ptr<ResultMultiVectorType> W1_; 

    /**
    * @brief Unique pointer to the second result multi-vector.
    * 
    * This unique pointer holds another result multi-vector for storing additional intermediate computation results.
    */
    std::unique_ptr<ResultMultiVectorType> W2_; 

    /**
    * @brief Unique pointer to the Ritz values matrix.
    * 
    * This unique pointer holds the Ritz values, which are used in the Chase algorithm's eigenvalue computation.
    */
    std::unique_ptr<chase::distMatrix::RedundantMatrix<chase::Base<T>>> ritzv_; 

    /**
    * @brief Unique pointer to the residuals matrix.
    * 
    * This unique pointer stores the residuals of the computed eigenvalues during the Chase algorithm.
    */
    std::unique_ptr<chase::distMatrix::RedundantMatrix<chase::Base<T>>> resid_; 

    /**
    * @brief Unique pointer to an auxiliary matrix.
    * 
    * This unique pointer holds an auxiliary matrix used in the intermediate steps of the Chase algorithm.
    */
    std::unique_ptr<chase::distMatrix::RedundantMatrix<T>> A_; 

    /**
    * @brief Configuration for the Chase algorithm.
    * 
    * This member holds the configuration settings for the Chase algorithm, such as parameters for 
    * matrix dimensions and other algorithmic settings.
    */
    chase::ChaseConfig<T> config_;
};

}
}
