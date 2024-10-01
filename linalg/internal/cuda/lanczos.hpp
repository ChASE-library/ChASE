#pragma once

#include "linalg/cublaspp/cublaspp.hpp"
#include "linalg/cusolverpp/cusolverpp.hpp"
#include "Impl/cuda/cuda_utils.hpp"
#include "linalg/matrix/matrix.hpp"
#include "linalg/internal/cuda/lacpy.hpp"

using namespace chase::linalg;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    template<typename T>
    void lanczos(cublasHandle_t cublas_handle, 
                 std::size_t M, 
                 std::size_t numvec,
                 chase::matrix::MatrixGPU<T>& H, 
                 chase::matrix::MatrixGPU<T>& V, 
                 chase::Base<T>* upperb, 
                 chase::Base<T>* ritzv, 
                 chase::Base<T>* Tau, 
                 chase::Base<T>* ritzV)
    {
        T One = T(1.0);
        T Zero = T(0.0);
        std::size_t N = H.rows();
        std::vector<chase::Base<T>> r_beta(numvec);
        
        std::vector<chase::Base<T>> d(M * numvec);
        std::vector<chase::Base<T>> e(M * numvec);

        std::vector<chase::Base<T>> real_alpha(numvec);
        std::vector<T> alpha(numvec, T(1.0));
        std::vector<T> beta(numvec, T(0.0));

        auto v_0 = chase::matrix::MatrixGPU<T>(N, numvec);
        auto v_1 = chase::matrix::MatrixGPU<T>(N, numvec);
        auto v_2 = chase::matrix::MatrixGPU<T>(N, numvec);
        chase::linalg::internal::cuda::t_lacpy('A', 
                                                N, 
                                                numvec, 
                                                V.gpu_data(), 
                                                V.gpu_ld(), 
                                                v_1.gpu_data(), 
                                                v_1.gpu_ld());

        for(auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(cublas_handle, 
                                                                    v_1.rows(), 
                                                                    v_1.gpu_data() + i * v_1.gpu_ld(),
                                                                    1, 
                                                                    &real_alpha[i]));
        }

        for(auto i = 0; i < numvec; i++)
        {
            alpha[i] = T(1 / real_alpha[i]);
        }

        for(auto i = 0; i < numvec; i++)
        {
              CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle, 
                                                                      v_1.rows(), 
                                                                      &alpha[i], 
                                                                      v_1.gpu_data() + i * v_1.gpu_ld(),
                                                                      1));  
        }

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            for(auto i = 0; i < numvec; i++){
                CHECK_CUDA_ERROR( cudaMemcpy(V.gpu_data() + k * V.gpu_ld(), 
                                             v_1.gpu_data() + i * v_1.gpu_ld(), 
                                             v_1.rows() * sizeof(T),
                                             cudaMemcpyDeviceToDevice ));
            }

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle,
                                                                          CUBLAS_OP_N,
                                                                          CUBLAS_OP_N,
                                                                          H.rows(),
                                                                          numvec,
                                                                          H.cols(),
                                                                          &One,
                                                                          H.gpu_data(),
                                                                          H.gpu_ld(),
                                                                          v_1.gpu_data(),
                                                                          v_1.gpu_ld(),
                                                                          &Zero,
                                                                          v_2.gpu_data(),
                                                                          v_2.gpu_ld()));


            for(auto i = 0; i < numvec; i++)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(cublas_handle, 
                                                                             v_1.rows(), 
                                                                             v_1.gpu_data() + i * v_1.gpu_ld(), 
                                                                             1, 
                                                                             v_2.gpu_data() + i * v_2.gpu_ld(), 
                                                                             1, 
                                                                             &alpha[i])); 

            }

            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = -alpha[i];
            }

            for(auto i = 0; i < numvec; i++)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle, 
                                                                              v_1.rows(), 
                                                                              &alpha[i], 
                                                                              v_1.gpu_data() + i * v_1.gpu_ld(), 
                                                                              1, 
                                                                              v_2.gpu_data() + i * v_2.gpu_ld(), 
                                                                              1));
            }
            
            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = -alpha[i];
            }

            for(auto i = 0; i < numvec; i++)
            {
                d[k + M * i] = std::real(alpha[i]);
            }

            if(k > 0){
                for(auto i = 0; i < numvec; i++)
                {
                    beta[i] = T(-r_beta[i]);
                }
                for(auto i = 0; i < numvec; i++)
                {
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle, 
                                                                                v_0.rows(), 
                                                                                &beta[i], 
                                                                                v_0.gpu_data() + i * v_0.gpu_ld(), 
                                                                                1, 
                                                                                v_2.gpu_data() + i * v_2.gpu_ld(), 
                                                                                1));                    
                }                                
            }

            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = -beta[i];
            }

            for(auto i = 0; i < numvec; i++)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(cublas_handle, 
                                                                        v_2.rows(), 
                                                                        v_2.gpu_data() + i * v_2.gpu_ld(),
                                                                        1, 
                                                                        &r_beta[i]));                
            }

            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = T(1 / r_beta[i]);
            }

            if (k == M - 1)
                break;

            for(auto i = 0; i < numvec; i++)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle, 
                                                                        v_2.rows(), 
                                                                        &beta[i], 
                                                                        v_2.gpu_data() + i * v_2.gpu_ld(),
                                                                        1));              
            }

            for(auto i = 0; i < numvec; i++)
            {
                e[k + M * i] = r_beta[i];
            }

            v_1.swap(v_0);
            v_1.swap(v_2);    
                                
        }

        chase::linalg::internal::cuda::t_lacpy('A', 
                                                N, 
                                                numvec, 
                                                v_1.gpu_data(),
                                                v_1.gpu_ld(),
                                                V.gpu_data(), 
                                                V.gpu_ld());                

        int notneeded_m;
        std::size_t vl, vu;
        Base<T> ul, ll;
        int tryrac = 0;
        std::vector<int> isuppz(2 * M);

        for(auto i = 0; i < numvec; i++)
        {
            lapackpp::t_stemr(LAPACK_COL_MAJOR, 'V', 'A', M, d.data() + i * M, e.data() + i * M, ul, ll, vl, vu,
                                &notneeded_m, ritzv + M * i, ritzV, M, M, isuppz.data(), &tryrac);
            for (std::size_t k = 0; k < M; ++k)
            {
                Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
            }
        }

        Base<T> max;
        *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
                  std::abs(r_beta[0]);

        for(auto i = 1; i < numvec; i++)
        {
          max = std::max(std::abs(ritzv[i * M]), std::abs(ritzv[ (i + 1) * M - 1])) +
                  std::abs(r_beta[i]);
          *upperb = std::max(max, *upperb);        
        }               
    }


    template<typename T>
    void lanczos(cublasHandle_t cublas_handle,
                 std::size_t M, 
                 chase::matrix::MatrixGPU<T>& H,
                 chase::matrix::MatrixGPU<T>& V, 
                 chase::Base<T>* upperb)
    {
        T One = T(1.0);
        T Zero = T(0.0);
        chase::Base<T> r_beta;
        std::size_t N = H.rows();

        std::vector<Base<T>> d(M);
        std::vector<Base<T>> e(M);

        chase::Base<T> real_alpha;
        T alpha = T(1.0);
        T beta = T(0.0);

        auto v_0 = chase::matrix::MatrixGPU<T>(N, 1);
        auto v_1 = chase::matrix::MatrixGPU<T>(N, 1);
        auto v_2 = chase::matrix::MatrixGPU<T>(N, 1);

        chase::linalg::internal::cuda::t_lacpy('A', 
                                                N, 
                                                1, 
                                                V.gpu_data(), 
                                                V.gpu_ld(), 
                                                v_1.gpu_data(), 
                                                v_1.gpu_ld());

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(cublas_handle, 
                                                                v_1.rows(), 
                                                                v_1.gpu_data(),
                                                                1, 
                                                                &real_alpha));        
        alpha = T(1 / real_alpha);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle, 
                                                                v_1.rows(), 
                                                                &alpha, 
                                                                v_1.gpu_data(),
                                                                1));          
        for (std::size_t k = 0; k < M; k = k + 1)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle,
                                                                          CUBLAS_OP_N,
                                                                          CUBLAS_OP_N,
                                                                          H.rows(),
                                                                          1,
                                                                          H.cols(),
                                                                          &One,
                                                                          H.gpu_data(),
                                                                          H.gpu_ld(),
                                                                          v_1.gpu_data(),
                                                                          v_1.gpu_ld(),
                                                                          &Zero,
                                                                          v_2.gpu_data(),
                                                                          v_2.gpu_ld()));

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(cublas_handle, 
                                                                            v_1.rows(), 
                                                                            v_1.gpu_data(), 
                                                                            1, 
                                                                            v_2.gpu_data(), 
                                                                            1, 
                                                                            &alpha)); 
            alpha = -alpha;

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle, 
                                                                            v_1.rows(), 
                                                                            &alpha, 
                                                                            v_1.gpu_data(), 
                                                                            1, 
                                                                            v_2.gpu_data(), 
                                                                            1));
            alpha = -alpha;

            d[k] = std::real(alpha);
            
            if(k > 0){
                beta = T(-r_beta);
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle, 
                                                                            v_0.rows(), 
                                                                            &beta, 
                                                                            v_0.gpu_data(), 
                                                                            1, 
                                                                            v_2.gpu_data(), 
                                                                            1));                                              
            }

            beta = -beta;

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(cublas_handle, 
                                                                    v_2.rows(), 
                                                                    v_2.gpu_data(),
                                                                    1, 
                                                                    &r_beta));  
            beta = T(1 / r_beta);
            
            if (k == M - 1)
                break;

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle, 
                                                                    v_2.rows(), 
                                                                    &beta, 
                                                                    v_2.gpu_data(),
                                                                    1));  
            e[k] = r_beta;

            v_1.swap(v_0);
            v_1.swap(v_2);                        
        }        

        int notneeded_m;
        std::size_t vl, vu;
        Base<T> ul, ll;
        int tryrac = 0;
        std::vector<int> isuppz(2 * M);
        std::vector<Base<T>> ritzv(M);

        lapackpp::t_stemr<Base<T>>(LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
                         &notneeded_m, ritzv.data(), NULL, M, M, isuppz.data(), &tryrac);

        *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
                  std::abs(r_beta);

    }
}
}
}
}