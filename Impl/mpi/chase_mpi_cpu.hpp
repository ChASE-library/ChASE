#pragma once

#include <cstring>
#include <memory>
#include <random>
#include <vector>
#include "algorithm/chaseBase.hpp"
#include "linalg/matrix/matrix.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "Impl/grid/mpiGrid2D.hpp"
#include "linalg/internal/mpi/cholqr.hpp"
#include "linalg/internal/mpi/lanczos.hpp"
#include "linalg/internal/mpi/residuals.hpp"
#include "linalg/internal/mpi/rayleighRitz.hpp"
#include "linalg/internal/mpi/shiftDiagonal.hpp"
#ifdef HAS_SCALAPACK
#include "linalg/scalapackpp/scalapackpp.hpp"
#endif
#include "linalg/internal/mpi/symOrHerm.hpp"
#include "algorithm/types.hpp"
#include "../../linalg/internal/typeTraits.hpp"

using namespace chase::linalg;

namespace chase
{
namespace Impl
{
template <class T, typename MatrixType>
class ChaseMPICPU : public ChaseBase<T>
{
public:
    ChaseMPICPU(std::size_t nev,
                std::size_t nex,
                MatrixType *H,
                chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column> *V,
                chase::Base<T> *ritzv
                ): nev_(nev), nex_(nex), nevex_(nev + nex), config_(H->g_rows(), nev, nex), N_(H->g_rows())
    {
        if(H->g_rows() != H->g_cols())
        {
            std::runtime_error("ChASE requires the matrix solved to be squared");
        }

        if( H->getMpiGrid() != V->getMpiGrid())
        {   
            std::runtime_error("ChASE requires the matrix and eigenvectors mapped to same MPI grid");
        }

        Hmat_ = H;
        V1_ = V;
        V2_ = new chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(Hmat_->g_rows(), nevex_, Hmat_->getMpiGrid_shared_ptr());
        W1_ = new chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(Hmat_->g_rows(), nevex_, Hmat_->getMpiGrid_shared_ptr());
        W2_ = new chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(Hmat_->g_rows(), nevex_, Hmat_->getMpiGrid_shared_ptr());
        ritzv_ = new chase::distMatrix::RedundantMatrix<chase::Base<T>>(nevex_, 1, nevex_, ritzv, Hmat_->getMpiGrid_shared_ptr());
        resid_ = new chase::distMatrix::RedundantMatrix<chase::Base<T>>(nevex_, 1, Hmat_->getMpiGrid_shared_ptr());
        A_ = new chase::distMatrix::RedundantMatrix<T>(nevex_, nevex_, Hmat_->getMpiGrid_shared_ptr());

        MPI_Comm_rank(Hmat_->getMpiGrid()->get_comm(), &my_rank_);
        MPI_Comm_size(Hmat_->getMpiGrid()->get_comm(), &nprocs_);
        coords_ = Hmat_->getMpiGrid()->get_coords();
        dims_ = Hmat_->getMpiGrid()->get_dims();

    }

    ChaseMPICPU(const ChaseMPICPU&) = delete;

    ~ChaseMPICPU() {}

    std::size_t GetN() const override { return N_; }

    std::size_t GetNev() override { return nev_; }
    
    std::size_t GetNex() override { return nex_; }

    chase::Base<T>* GetRitzv() override { return ritzv_->l_data(); }
    chase::Base<T>* GetResid() override { return resid_->l_data(); }
    ChaseConfig<T>& GetConfig() override { return config_; }
    int get_nprocs() override { return nprocs_; }

    void loadProblemFromFile(std::string filename)
    {
        Hmat_->readFromBinaryFile(filename);
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
        is_sym_ = chase::linalg::internal::mpi::checkSymmetryEasy(*Hmat_);  
        return is_sym_;
    }

    bool isSym() { return is_sym_; }

    void symOrHermMatrix(char uplo) override
    {
        chase::linalg::internal::mpi::symOrHermMatrix(uplo, *Hmat_);   
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
        chase::linalg::internal::mpi::lanczos(m, 
                                              *Hmat_, 
                                              *V1_, 
                                              upperb);
    }

    void Lanczos(std::size_t M, std::size_t numvec, chase::Base<T>* upperb,
                         chase::Base<T>* ritzv, chase::Base<T>* Tau, chase::Base<T>* ritzV) override
    {
        chase::linalg::internal::mpi::lanczos(M, 
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
        chase::linalg::internal::mpi::shiftDiagonal(*Hmat_, c);

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

                if (next_ == NextOp::bAc)
                {
                    chase::linalg::internal::mpi::MatrixMultiplyMultiVectors<singlePrecisionT>(&alpha_sp, 
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
                    chase::linalg::internal::mpi::MatrixMultiplyMultiVectors<singlePrecisionT>(&alpha_sp, 
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
                    chase::linalg::internal::mpi::MatrixMultiplyMultiVectors(&alpha, 
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
                    chase::linalg::internal::mpi::MatrixMultiplyMultiVectors(&alpha, 
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
                chase::linalg::internal::mpi::MatrixMultiplyMultiVectors(&alpha, 
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
                chase::linalg::internal::mpi::MatrixMultiplyMultiVectors(&alpha, 
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

        if (disable == 1)
        {
#ifdef HAS_SCALAPACK
            chase::linalg::internal::mpi::houseHoulderQR(*V1_);
#else
        std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
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
                info = chase::linalg::internal::mpi::shiftedcholQR2(V1_->g_rows(),
                                                                    V1_->l_rows(), 
                                                                    V1_->l_cols(), 
                                                                    V1_->l_data(),  
                                                                    V1_->l_ld(), 
                                                                    V1_->getMpiGrid()->get_col_comm(), 
                                                                    A_->l_data());
            }
            else if(cond < cond_threshold_lower)
            {
                info = chase::linalg::internal::mpi::cholQR1(V1_->l_rows(), 
                                                             V1_->l_cols(), 
                                                             V1_->l_data(),  
                                                             V1_->l_ld(), 
                                                             V1_->getMpiGrid()->get_col_comm());
            }
            else
            {
                info = chase::linalg::internal::mpi::cholQR2(V1_->l_rows(), 
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
                chase::linalg::internal::mpi::houseHoulderQR(*V1_);
#else
                std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
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
        chase::linalg::internal::mpi::rayleighRitz(*Hmat_, 
                                                   *V1_, 
                                                   *V2_, 
                                                   *W1_, 
                                                   *W2_, 
                                                   ritzv_->l_data(), 
                                                   locked_, 
                                                   block,
                                                   A_);

        chase::linalg::lapackpp::t_lacpy('A',
                                         V2_->l_rows(),
                                         block,
                                         V1_->l_data() + locked_ * V1_->l_ld(),
                                         V1_->l_ld(),
                                         V2_->l_data() + locked_ * V2_->l_ld(),
                                         V2_->l_ld());   



    }

    void Resd(chase::Base<T>* ritzv, chase::Base<T>* resd, std::size_t fixednev) override 
    {
        chase::linalg::internal::mpi::residuals(*Hmat_,
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
    enum NextOp
    {
        cAb,
        bAc
    };
    NextOp next_; 

    bool is_sym_;
    std::size_t nev_;
    std::size_t nex_;
    std::size_t nevex_;
    std::size_t locked_;

    std::size_t N_;

    int nprocs_;
    int my_rank_;
    int *coords_;
    int *dims_;

    chase::distMatrix::BlockBlockMatrix<T> *Hmat_;
    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column> *V1_;
    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column> *V2_;
    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row> *W1_;
    chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row> *W2_;

    chase::distMatrix::RedundantMatrix<chase::Base<T>> *ritzv_;
    chase::distMatrix::RedundantMatrix<chase::Base<T>> *resid_;
    chase::distMatrix::RedundantMatrix<T> *A_;

    chase::ChaseConfig<T> config_;
};

}
}