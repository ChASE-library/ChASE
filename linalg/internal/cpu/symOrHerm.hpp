#pragma once

#include <vector>
#include <random>
#include "algorithm/types.hpp"
#include "linalg/blaspp/blaspp.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cpu
{
    template<typename T>
    bool checkSymmetryEasy(std::size_t N, T *H, std::size_t ldh) 
    {       
        std::vector<T> v(N);
        std::vector<T> u(N);
        std::vector<T> uT(N);

        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;
        for (auto i = 0; i < N; i++)
        {
            v[i] = getRandomT<T>([&]() { return d(gen); });
        }
        
        T One = T(1.0);
        T Zero = T(0.0);

        chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                     CblasNoTrans, 
                                     CblasNoTrans, 
                                     N, 
                                     1, 
                                     N,                                 
                                     &One,                                      
                                     H, 
                                     ldh,                                  
                                     v.data(), 
                                     N,                             
                                     &Zero,                                     
                                     u.data(), 
                                     N);

        chase::linalg::blaspp::t_gemm(CblasColMajor, 
                                      CblasConjTrans, 
                                      CblasNoTrans, 
                                      N, 
                                      1, 
                                      N,                                
                                      &One,                                      
                                      H, 
                                      ldh,                                  
                                      v.data(), 
                                      N,                             
                                      &Zero,                                     
                                      uT.data(), 
                                      N);

        bool is_sym = true;
        for(auto i = 0; i < N; i++)
        {
            if(!(u[i] == uT[i]))
            {
                is_sym = false;
                return is_sym;
            }
        }

        return is_sym;
        
    }

    template<typename T>
    void symOrHermMatrix(char uplo, std::size_t N, T *H, std::size_t ldh) 
    {
        
        if(uplo == 'U')
        {
            for(auto j = 0; j < N; j++)
            {
                for(auto i = 0; i < j; i++)
                {
                    H[j + i * ldh]= conjugate(H[i + j * ldh]);
                }
            }
        }else
        {
            for(auto i = 0; i < N; i++)
            {
                for(auto j = 0; j < i; j++)
                {
                    H[j + i * ldh]= conjugate(H[i + j * ldh]);
                }
            }
        }
      
    }
}
}
}
}