#pragma once

#include <cstring>
#include <memory>
#include <random>
#include <vector>
#include "algorithm/chaseBase.hpp"
#include "linalg/matrix/matrix.hpp"
#include "linalg/internal/cuda/cholqr.hpp"
#include "linalg/internal/cuda/lanczos.hpp"
#include "linalg/internal/cuda/shiftDiagonal.hpp"
#include "linalg/internal/cuda/residuals.hpp"
#include "linalg/internal/cuda/rayleighRitz.hpp"
#include "linalg/internal/cuda/lacpy.hpp"
#include "linalg/internal/cpu/symOrHerm.hpp"
#include "linalg/internal/cuda/random_normal_distribution.hpp"

#include "algorithm/types.hpp"

using namespace chase::linalg;

namespace chase
{
namespace Impl
{
template <class T>
class ChaseGPUSeq : public ChaseBase<T>
{
public:
    ChaseGPUSeq(std::size_t N,
                std::size_t nev,
                std::size_t nex,
                T *H,
                std::size_t ldh,
                T *V1,
                std::size_t ldv,
                chase::Base<T> *ritzv)
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
        Hmat_ = chase::matrix::MatrixGPU<T>(N_, N_, ldh_, H_);
        Vec1_ = chase::matrix::MatrixGPU<T>(N_, nevex_, ldv_, V1_);
        Vec2_ = chase::matrix::MatrixGPU<T>(N_, nevex_);
        resid_ = chase::matrix::MatrixGPU<chase::Base<T>>(nevex_, 1);
        ritzvs_ = chase::matrix::MatrixGPU<chase::Base<T>>(nevex_, 1, nevex_, ritzv_);
        A_ = chase::matrix::MatrixGPU<T>(nevex_, nevex_);

        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));
        CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH_));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, stream_));
        CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolverH_, stream_));

        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo_, sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_return_, sizeof(T) * nevex_));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&tmp_, N_ * sizeof(T)));

        int lwork_geqrf = 0;
        int lwork_orgqr = 0;

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgeqrf_bufferSize(
                                                            cusolverH_, 
                                                            N_, 
                                                            nevex_, 
                                                            Vec1_.gpu_data(), 
                                                            Vec1_.gpu_ld(), 
                                                            &lwork_geqrf));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgqr_bufferSize(
                                                            cusolverH_, 
                                                            N_, 
                                                            nevex_, 
                                                            nevex_,
                                                            Vec1_.gpu_data(), 
                                                            Vec1_.gpu_ld(),  
                                                            d_return_, 
                                                            &lwork_orgqr));

        lwork_ = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;

        int lwork_heevd = 0;

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                                                            cusolverH_, 
                                                            CUSOLVER_EIG_MODE_VECTOR, 
                                                            CUBLAS_FILL_MODE_LOWER,
                                                            nevex_, 
                                                            A_.gpu_data(), 
                                                            A_.gpu_ld(), 
                                                            ritzvs_.gpu_data(), 
                                                            &lwork_heevd));
        if (lwork_heevd > lwork_)
        {
            lwork_ = lwork_heevd;
        }

        int lwork_potrf = 0;

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                            cusolverH_, 
                                                            CUBLAS_FILL_MODE_UPPER, 
                                                            nevex_, 
                                                            A_.gpu_data(), 
                                                            A_.gpu_ld(),
                                                            &lwork_potrf));
        if (lwork_potrf > lwork_)
        {
            lwork_ = lwork_potrf;
        }
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_work_, sizeof(T) * lwork_));

    }

    ChaseGPUSeq(const ChaseGPUSeq&) = delete;

    ~ChaseGPUSeq() 
    {
        if (cublasH_)
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH_));
        if (cusolverH_)
            CHECK_CUSOLVER_ERROR(cusolverDnDestroy(cusolverH_));
        if (d_work_)
            CHECK_CUDA_ERROR(cudaFree(d_work_));
        if (devInfo_)
            CHECK_CUDA_ERROR(cudaFree(devInfo_));
        if (d_return_)
            CHECK_CUDA_ERROR(cudaFree(d_return_));
        if (tmp_)
            CHECK_CUDA_ERROR(cudaFree(tmp_));

    }

    std::size_t GetN() const override { return N_; }

    std::size_t GetNev() override { return nev_; }
    
    std::size_t GetNex() override { return nex_; }

    chase::Base<T>* GetRitzv() override { return ritzv_; }
    chase::Base<T>* GetResid() override { resid_.allocate_cpu_data(); return resid_.cpu_data(); }
    ChaseConfig<T>& GetConfig() override { return config_; }
    int get_nprocs() override { return 1; }

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
        is_sym_ = chase::linalg::internal::cpu::checkSymmetryEasy(N_, Hmat_.cpu_data(), Hmat_.cpu_ld());  
        return is_sym_;
    }

    bool isSym() { return is_sym_; }

    void symOrHermMatrix(char uplo) override
    {
        chase::linalg::internal::cpu::symOrHermMatrix(uplo, N_, Hmat_.cpu_data(), Hmat_.cpu_ld());
    }

    void Start() override
    {
        locked_ = 0;
    }

    void initVecs(bool random) override
    {         
        if (random)
        {
            chase::linalg::internal::cuda::init_random_vectors(Vec1_.gpu_data(), Vec1_.gpu_ld() * Vec1_.cols());
        }

        chase::linalg::internal::cuda::t_lacpy('A', 
                                          Vec1_.rows(), 
                                          Vec1_.cols(), 
                                          Vec1_.gpu_data(), 
                                          Vec1_.gpu_ld(),
                                          Vec2_.gpu_data(), 
                                          Vec2_.gpu_ld());  
        
        Hmat_.H2D();
    }

    void Lanczos(std::size_t M, chase::Base<T>* upperb) override
    {
        chase::linalg::internal::cuda::lanczos(cublasH_,
                                               M, 
                                               Hmat_,
                                               Vec1_,
                                               upperb);    
    }

    void Lanczos(std::size_t M, std::size_t numvec, chase::Base<T>* upperb,
                         chase::Base<T>* ritzv, chase::Base<T>* Tau, chase::Base<T>* ritzV) override
    {    
        chase::linalg::internal::cuda::lanczos(cublasH_,
                                               M, 
                                               numvec,
                                               Hmat_,
                                               Vec1_,
                                               upperb,
                                               ritzv,
                                               Tau,
                                               ritzV);
    }

    void LanczosDos(std::size_t idx, std::size_t m, T* ritzVc) override
    {
        T alpha = T(1.0);
        T beta = T(0.0);
        
        CHECK_CUBLAS_ERROR(cublasSetMatrix(m, 
                                        idx, 
                                        sizeof(T), 
                                        ritzVc, 
                                        m, 
                                        A_.gpu_data(), 
                                        A_.gpu_ld()));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublasH_, 
                                                                CUBLAS_OP_N, 
                                                                CUBLAS_OP_N, 
                                                                Vec1_.rows(), 
                                                                idx, 
                                                                m, 
                                                                &alpha,
                                                                Vec1_.gpu_data(), 
                                                                Vec1_.gpu_ld(), 
                                                                A_.gpu_data(), 
                                                                A_.gpu_ld(), 
                                                                &beta, 
                                                                Vec2_.gpu_data(), 
                                                                Vec2_.gpu_ld()));
        
        chase::linalg::internal::cuda::t_lacpy('A', 
                                    Vec2_.rows(), 
                                    m, 
                                    Vec2_.gpu_data(), 
                                    Vec2_.gpu_ld(),
                                    Vec1_.gpu_data(), 
                                    Vec1_.gpu_ld());  

    }

    void Shift(T c, bool isunshift = false) override
    {          
        chase::linalg::internal::cuda::shiftDiagonal(Hmat_, std::real(c));

    }
    
    void HEMM(std::size_t block, T alpha, T beta, std::size_t offset) override
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublasH_,
                                                                CUBLAS_OP_N,
                                                                CUBLAS_OP_N,
                                                                Hmat_.rows(),
                                                                block,
                                                                Hmat_.cols(),
                                                                &alpha,
                                                                Hmat_.gpu_data(),
                                                                Hmat_.gpu_ld(),
                                                                Vec1_.gpu_data() + offset * Vec1_.gpu_ld() + locked_ * Vec1_.gpu_ld(),
                                                                Vec1_.gpu_ld(),
                                                                &beta,
                                                                Vec2_.gpu_data() + offset * Vec2_.gpu_ld() + locked_ * Vec2_.gpu_ld(),
                                                                Vec2_.gpu_ld()));

        Vec1_.swap(Vec2_);
    }

    void QR(std::size_t fixednev, chase::Base<T> cond) override
    {
        chase::linalg::internal::cuda::t_lacpy('A', 
                                                Vec2_.rows(), 
                                                locked_, 
                                                Vec1_.gpu_data(), 
                                                Vec1_.gpu_ld(),
                                                Vec2_.gpu_data(), 
                                                Vec2_.gpu_ld());   

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
            chase::linalg::internal::cuda::houseHoulderQR(cusolverH_,
                                                        Vec1_,
                                                        d_return_,
                                                        devInfo_,
                                                        d_work_,
                                                        lwork_);
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
                
                info = chase::linalg::internal::cuda::shiftedcholQR2(cublasH_,
                                                              cusolverH_,
                                                              Vec1_,
                                                              d_work_,
                                                              lwork_,
                                                              &A_);
            }
            else if(cond < cond_threshold_lower)
            {
                info = chase::linalg::internal::cuda::cholQR1(cublasH_,
                                                              cusolverH_,
                                                              Vec1_,
                                                              d_work_,
                                                              lwork_,
                                                              &A_); 
            }
            else
            {
                info = chase::linalg::internal::cuda::cholQR2(cublasH_,
                                                              cusolverH_,
                                                              Vec1_,
                                                              d_work_,
                                                              lwork_,
                                                              &A_);
  
            }

            if (info != 0)
            {
#ifdef CHASE_OUTPUT
                std::cout << "CholeskyQR doesn't work, Househoulder QR will be used." << std::endl;
#endif
                chase::linalg::internal::cuda::houseHoulderQR(cusolverH_,
                                                            Vec1_,
                                                            d_return_,
                                                            devInfo_,
                                                            d_work_,
                                                            lwork_);
            }
        }

        chase::linalg::internal::cuda::t_lacpy('A', 
                                                Vec1_.rows(), 
                                                locked_, 
                                                Vec2_.gpu_data(), 
                                                Vec2_.gpu_ld(),
                                                Vec1_.gpu_data(), 
                                                Vec1_.gpu_ld());    
    }

    void RR(chase::Base<T>* ritzv, std::size_t block) override
    {
        std::size_t locked = (nev_ + nex_) - block;
        chase::linalg::internal::cuda::rayleighRitz(cublasH_,
                                                    cusolverH_,
                                                    Hmat_,
                                                    Vec1_,
                                                    Vec2_,
                                                    ritzvs_,
                                                    locked,
                                                    block,
                                                    devInfo_,
                                                    d_work_,
                                                    lwork_,
                                                    &A_);
        Vec1_.swap(Vec2_);
    }

    void Resd(chase::Base<T>* ritzv, chase::Base<T>* resd, std::size_t fixednev) override
    {
        std::size_t unconverged = (nev_ + nex_) - fixednev;
        chase::linalg::internal::cuda::residuals(cublasH_,
                                                 Hmat_,
                                                 Vec1_,
                                                 ritzvs_.gpu_data(),
                                                 resid_.gpu_data(),
                                                 fixednev,
                                                 unconverged,
                                                 &Vec2_);
        CHECK_CUDA_ERROR(cudaMemcpy(resd, 
                                    resid_.gpu_data() + fixednev, 
                                    unconverged * sizeof(chase::Base<T>),
                                    cudaMemcpyDeviceToHost));      
    }

    void Swap(std::size_t i, std::size_t j) override
    {
        chase::linalg::internal::cuda::t_lacpy('A',
                                               Vec1_.rows(),
                                               1,
                                               Vec1_.gpu_data() + i * Vec1_.gpu_ld(),
                                               Vec1_.gpu_ld(),
                                               tmp_,
                                               N_);
        chase::linalg::internal::cuda::t_lacpy('A',
                                               Vec1_.rows(),
                                               1,
                                               Vec1_.gpu_data() + j * Vec1_.gpu_ld(),
                                               Vec1_.gpu_ld(),
                                               Vec1_.gpu_data() + i * Vec1_.gpu_ld(),
                                               Vec1_.gpu_ld());
        chase::linalg::internal::cuda::t_lacpy('A',
                                               Vec1_.rows(),
                                               1,
                                               tmp_,
                                               N_,
                                               Vec1_.gpu_data() + j * Vec1_.gpu_ld(),
                                               Vec1_.gpu_ld());                                       
    }

    void Lock(std::size_t new_converged) override
    {
        locked_ += new_converged;
    }
 
    void End() override 
    {
        Vec1_.D2H();
    }
        
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
    T *tmp_;

    chase::matrix::MatrixGPU<T> Hmat_;
    chase::matrix::MatrixGPU<T> Vec1_;
    chase::matrix::MatrixGPU<T> Vec2_;
    chase::matrix::MatrixGPU<T> A_;
    chase::matrix::MatrixGPU<chase::Base<T>> ritzvs_;
    chase::matrix::MatrixGPU<chase::Base<T>> resid_;
    chase::ChaseConfig<T> config_;

    cudaStream_t stream_; 
    cublasHandle_t cublasH_;      
    cusolverDnHandle_t cusolverH_;


    int* devInfo_;
    T* d_return_;
    T* d_work_;
    int lwork_ = 0;
}; 

}    
}