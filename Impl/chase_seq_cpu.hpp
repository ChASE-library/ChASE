#pragma once

#include <cstring>
#include <memory>
#include <random>
#include <vector>
#include "algorithm/chaseBase.hpp"
#include "linalg/matrix/matrix.hpp"
#include "linalg/internal/cpu/cholqr1.hpp"
#include "linalg/internal/cpu/lanczos.hpp"
#include "linalg/internal/cpu/residuals.hpp"
#include "linalg/internal/cpu/rayleighRitz.hpp"
#include "linalg/internal/cpu/symOrHerm.hpp"
#include "algorithm/types.hpp"

using namespace chase::linalg;

namespace chase
{
namespace Impl
{
template <class T>
class ChaseCPUSeq : public ChaseBase<T>
{
public:
    ChaseCPUSeq(std::size_t N, 
                std::size_t nex, 
                std::size_t nev, 
                T* H, 
                std::size_t ldh, 
                T* V1, 
                std::size_t ldv,
                chase::Base<T>* ritzv)
                : N_(N), 
                  H_(H),
                  V1_(V1),
                  ldh_(ldh),
                  ldv_(ldv),
                  ritzv_(ritzv),
                  nev_(nev), 
                  nex_(nex), 
                  nevex_(nev+nex),
                  config_(N, nev, nex)
    {
        Hmat_ = chase::matrix::MatrixCPU<T>(N_, N_, ldh_, H_);
        Vec1_ = chase::matrix::MatrixCPU<T>(N_, nevex_, ldv_, V1_);
        Vec2_ = chase::matrix::MatrixCPU<T>(N_, nevex_);
        resid_ = chase::matrix::MatrixCPU<chase::Base<T>>(nevex_, 1);
        ritzvs_ = chase::matrix::MatrixCPU<chase::Base<T>>(nevex_, 1, nevex_, ritzv_);
        A_ = chase::matrix::MatrixCPU<T>(nevex_, nevex_);
    }

    ChaseCPUSeq(const ChaseCPUSeq&) = delete;

    ~ChaseCPUSeq() {}

    std::size_t GetN() const override {return N_;}

    std::size_t GetNev() override {return nev_;}
    
    std::size_t GetNex() override {return nex_;}

    chase::Base<T>* GetRitzv() override {return ritzvs_.data(); }
    chase::Base<T>* GetResid() override {return resid_.data(); }
    ChaseConfig<T>& GetConfig() override {return config_; }
    int get_nprocs() override {return 1;}

    void loadProblemFromFile(std::string filename)
    {
        Hmat_.readFromBinaryFile(filename);
    }

#ifdef CHASE_OUTPUT
    //! Print some intermediate infos during the solving procedure
    void Output(std::string str) override
    {
        std::cout << str;
    }
#endif

    bool checkSymmetryEasy() override
    {
        is_sym_ = chase::linalg::internal::cpu::checkSymmetryEasy(N_, Hmat_.data(), Hmat_.ld());  
        return is_sym_ = true;
    }

    bool isSym() {return is_sym_;}

    void symOrHermMatrix(char uplo) override
    {
        chase::linalg::internal::cpu::symOrHermMatrix(uplo, N_, Hmat_.data(), Hmat_.ld());
    }

    void Start() override
    {
        locked_ = 0;
    }

    void initVecs(bool random) override
    {

        if (random)
        {
            std::mt19937 gen(1337.0);
            std::normal_distribution<> d;
            for (auto j = 0; j < Vec1_.cols(); j++)
            {
                for (auto i = 0; i < Vec1_.rows(); i++)
                {
                    Vec1_.data()[i + j * Vec1_.ld()] = getRandomT<T>([&]() { return d(gen); });
                }
            }
        }

        chase::linalg::lapackpp::t_lacpy('A', 
                                          Vec1_.rows(), 
                                          Vec1_.cols(), 
                                          Vec1_.data(), 
                                          Vec1_.ld(),
                                          Vec2_.data(), 
                                          Vec2_.ld());     
    }

    void Lanczos(std::size_t m, chase::Base<T>* upperb) override
    {
        chase::linalg::internal::cpu::lanczos(m, 
                                              Hmat_.rows(),
                                              Hmat_.data(), 
                                              Hmat_.ld(), 
                                              Vec1_.data(), 
                                              Vec1_.ld(), 
                                              upperb);    
    }

    void Lanczos(std::size_t M, std::size_t numvec, chase::Base<T>* upperb,
                         chase::Base<T>* ritzv, chase::Base<T>* Tau, chase::Base<T>* ritzV) override
    {
        chase::linalg::internal::cpu::lanczos(M, 
                                              numvec, 
                                              Hmat_.rows(), 
                                              Hmat_.data(), 
                                              Hmat_.ld(), 
                                              Vec1_.data(), 
                                              Vec1_.ld(), 
                                              upperb, 
                                              ritzv, 
                                              Tau,  
                                              ritzV);       
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        
        T alpha = T(1.0);
        T beta = T(0.0);

        chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                      CblasNoTrans, 
                                      CblasNoTrans,
                                      N_, 
                                      idx, 
                                      m, 
                                      &alpha,
                                      Vec1_.data(), 
                                      Vec1_.ld(), 
                                      ritzVc, 
                                      m, 
                                      &beta, 
                                      Vec2_.data(), 
                                      Vec2_.ld());

        chase::linalg::lapackpp::t_lacpy('A', 
                                          Vec1_.rows(), 
                                          m, 
                                          Vec2_.data(), 
                                          Vec2_.ld(),
                                          Vec1_.data(), 
                                          Vec1_.ld());
    }

    void Shift(T c, bool isunshift = false) override
    {
        for (auto i = 0; i < N_; ++i)
        {
            Hmat_.data()[i + i * Hmat_.ld()] += c;
        }
#ifdef ENABLE_MIXED_PRECISION
        //mixed precision
        if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value)
        {
            auto min = *std::min_element(resid_.data() + locked_, resid_.data() + nev_);
            
            if(min > 1e-3)
            {
                if(isunshift)
                {
                    if(Hmat_.isSinglePrecisionEnabled())
                    {
                        Hmat_.disableSinglePrecision();
                    }
                    if(Vec1_.isSinglePrecisionEnabled())
                    {
                        Vec1_.disableSinglePrecision(true);
                    }   
                    if(Vec2_.isSinglePrecisionEnabled())
                    {
                        Vec2_.disableSinglePrecision();
                    }                                       
                }else
                {
                    std::cout << "Enable Single Precision in Filter" << std::endl;
                    Hmat_.enableSinglePrecision();
                    Vec1_.enableSinglePrecision();
                    Vec2_.enableSinglePrecision();  
                }
  
            }else
            {
                if(Hmat_.isSinglePrecisionEnabled())
                {
                    Hmat_.disableSinglePrecision();
                }
                if(Vec1_.isSinglePrecisionEnabled())
                {
                    Vec1_.disableSinglePrecision(true);
                }   
                if(Vec2_.isSinglePrecisionEnabled())
                {
                    Vec2_.disableSinglePrecision();
                }  
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
            auto min = *std::min_element(resid_.data() + locked_, resid_.data() + nev_);
            if(min > 1e-3)
            {
                auto Hmat_sp = Hmat_.matrix_sp();
                auto Vec1_sp = Vec1_.matrix_sp();
                auto Vec2_sp = Vec2_.matrix_sp();
                singlePrecisionT alpha_sp = static_cast<singlePrecisionT>(alpha);
                singlePrecisionT beta_sp = static_cast<singlePrecisionT>(beta);

                chase::linalg::blaspp::t_gemm<singlePrecisionT>(CblasColMajor, 
                                            CblasNoTrans, 
                                            CblasNoTrans, 
                                            Hmat_sp->rows(),
                                            block, 
                                            Hmat_sp->cols(), 
                                            &alpha_sp, 
                                            Hmat_sp->data(), 
                                            Hmat_sp->ld(),
                                            Vec1_sp->data() + offset * N_ + locked_ * N_, 
                                            Vec1_sp->ld(), 
                                            &beta_sp,
                                            Vec2_sp->data() + offset * N_ + locked_ * N_,
                                            Vec2_sp->ld());
            }
            else
            {
                chase::linalg::blaspp::t_gemm<T>(CblasColMajor, 
                                            CblasNoTrans, 
                                            CblasNoTrans, 
                                            Hmat_.rows(),
                                            block, 
                                            Hmat_.cols(), 
                                            &alpha, 
                                            Hmat_.data(), 
                                            Hmat_.ld(),
                                            Vec1_.data() + offset * N_ + locked_ * N_, 
                                            Vec1_.ld(), 
                                            &beta,
                                            Vec2_.data() + offset * N_ + locked_ * N_,
                                            Vec2_.ld());
            }
        }
        else
#endif        
        {        
            chase::linalg::blaspp::t_gemm<T>(CblasColMajor, 
                                        CblasNoTrans, 
                                        CblasNoTrans, 
                                        Hmat_.rows(),
                                        block, 
                                        Hmat_.cols(), 
                                        &alpha, 
                                        Hmat_.data(), 
                                        Hmat_.ld(),
                                        Vec1_.data() + offset * N_ + locked_ * N_, 
                                        Vec1_.ld(), 
                                        &beta,
                                        Vec2_.data() + offset * N_ + locked_ * N_,
                                        Vec2_.ld());
        }                              

        Vec1_.swap(Vec2_);
    }

    void QR(std::size_t fixednev, chase::Base<T> cond) override
    {
        
        chase::linalg::lapackpp::t_lacpy('A', 
                                          Vec2_.rows(), 
                                          locked_, 
                                          Vec1_.data(), 
                                          Vec1_.ld(),
                                          Vec2_.data(), 
                                          Vec2_.ld());   


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
            chase::linalg::internal::cpu::houseHoulderQR(Vec1_.rows(), 
                                                         Vec1_.cols(), 
                                                         Vec1_.data(), 
                                                         Vec1_.ld());
        }
        else
        {
#ifdef CHASE_OUTPUT
        std::cout << std::setprecision(2) << "cond(V): " << cond << std::endl;
#endif
            //if (display_bounds != 0)
            //{
            //  dla_->estimated_cond_evaluator(locked_, cond);
            //}
            int info = 1;

            if (cond > cond_threshold_upper)
            {
                info = chase::linalg::internal::cpu::shiftedcholQR2(Vec1_.rows(), 
                                                                    Vec1_.cols(), 
                                                                    Vec1_.data(), 
                                                                    Vec1_.ld(),
                                                                    A_.data());
            }
            else if(cond < cond_threshold_lower)
            {
                info = chase::linalg::internal::cpu::cholQR1(Vec1_.rows(), 
                                                             Vec1_.cols(), 
                                                             Vec1_.data(), 
                                                             Vec1_.ld(),
                                                             A_.data());   
            }
            else
            {
                info = chase::linalg::internal::cpu::cholQR2(Vec1_.rows(), 
                                                             Vec1_.cols(), 
                                                             Vec1_.data(), 
                                                             Vec1_.ld(),
                                                             A_.data());   
            }

            if (info != 0)
            {
#ifdef CHASE_OUTPUT
                std::cout << "CholeskyQR doesn't work, Househoulder QR will be used." << std::endl;
#endif
                chase::linalg::internal::cpu::houseHoulderQR(Vec1_.rows(), 
                                                             Vec1_.cols(), 
                                                             Vec1_.data(), 
                                                             Vec1_.ld());
            }
        }
        
        chase::linalg::lapackpp::t_lacpy('A', 
                                          Vec1_.rows(), 
                                          locked_, 
                                          Vec2_.data(), 
                                          Vec2_.ld(),
                                          Vec1_.data(), 
                                          Vec1_.ld());    
    }

    void RR(chase::Base<T>* ritzv, std::size_t block) override
    {

        std::size_t locked = (nev_ + nex_) - block;
     
        chase::linalg::internal::cpu::rayleighRitz(Hmat_.rows(),
                                                   Hmat_.data(),
                                                   Hmat_.ld(),
                                                   block, 
                                                   Vec1_.data() + locked_ * N_,
                                                   Vec1_.ld(),
                                                   Vec2_.data() + locked_ * N_,
                                                   Vec2_.ld(),
                                                   ritzvs_.data() + locked_,
                                                   A_.data()
                                                  );

        Vec1_.swap(Vec2_);
    }

    void Resd(chase::Base<T>* ritzv, chase::Base<T>* resd, std::size_t fixednev) override
    {
        std::size_t unconverged = (nev_ + nex_) - fixednev;

        chase::linalg::internal::cpu::residuals(Hmat_.rows(),
                                                Hmat_.data(),
                                                Hmat_.ld(),
                                                unconverged,
                                                ritzv,
                                                Vec1_.data() + fixednev * N_,
                                                Vec1_.ld(),
                                                resd,
                                                Vec2_.data());
    }

    void Swap(std::size_t i, std::size_t j) override
    {
        std::vector<T> tmp(Vec1_.rows());

        memcpy(tmp.data(), Vec1_.data() + Vec1_.ld() * i, Vec1_.rows() * sizeof(T));
        memcpy(Vec1_.data() + Vec1_.ld() * i, Vec1_.data() + Vec1_.ld() * j, Vec1_.rows() * sizeof(T));
        memcpy(Vec1_.data() + Vec1_.ld() * j, tmp.data(), Vec1_.rows() * sizeof(T));
    }

    void Lock(std::size_t new_converged) override
    {
        locked_ += new_converged;
    }
 
    void End() override { }
        
private:
    std::size_t N_;
    std::size_t locked_;
    std::size_t nev_;
    std::size_t nex_;
    std::size_t nevex_;
    std::size_t ldh_;
    std::size_t ldv_;

    bool is_sym_;

    T *H_;
    T *V1_;
    chase::Base<T> *ritzv_;

    chase::matrix::MatrixCPU<T> Hmat_;
    chase::matrix::MatrixCPU<T> Vec1_;
    chase::matrix::MatrixCPU<T> Vec2_;
    chase::matrix::MatrixCPU<T> A_;
    chase::matrix::MatrixCPU<chase::Base<T>> ritzvs_;
    chase::matrix::MatrixCPU<chase::Base<T>> resid_;
    chase::ChaseConfig<T> config_;

}; 

}    
}