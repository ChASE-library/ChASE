#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include "cublas_v2.h"
#include <cuda_runtime.h> 
#include <magma.h> 

#include <stdio.h>

#define DEBUG 1
#ifndef DEBUG
#define DEBUG 0
#endif
#define DEBUG_PRINT(s) if(DEBUG) printf(s);

typedef struct gpu_data{
   cuDoubleComplex * H_d;
   cuComplex * H_sp_d;
   cuComplex * V_sp_d;
   cuComplex * W_sp_d;
   cuDoubleComplex * V_d;
   cuDoubleComplex * W_d;
   cuDoubleComplex * tmp0_d;
   cuDoubleComplex * H1_d;
   cuDoubleComplex * V1_d;
   cuDoubleComplex * W1_d;
   cuDoubleComplex * tmp1_d;
   cuDoubleComplex * tau_d;
   double* ritzv_d;
   size_t sizeA, sizeV, sizeW;
   size_t sizeA_sp, sizeV_sp, sizeW_sp;

   cublasHandle_t handle, handle1;
   cudaStream_t stream0, stream1;
   cudaEvent_t event0, event1;


  int num_gpus; 
  int work_per_gpu[2];
} gpu_data_t;


static void handleError( cudaError_t error, const char *file, int line ) {
    if (error != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( error ), file, line );
        exit( -1 );
    }
}


static void handleError_cublas( cublasStatus_t error, const char *file, int line ) {
    if (error != CUBLAS_STATUS_SUCCESS) {
       if(error == CUBLAS_STATUS_INTERNAL_ERROR)
        printf( "CUBLAS_STATUS_INTERNAL_ERROR in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_EXECUTION_FAILED)
        printf( "CUBLAS_STATUS_EXECUTION_FAILED in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_MAPPING_ERROR)
        printf( "CUBLAS_STATUS_MAPPING_ERROR in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_ARCH_MISMATCH)
        printf( "CUBLAS_STATUS_ARCH_MISMATCH in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_INVALID_VALUE)
        printf( "CUBLAS_STATUS_INVALID_VALUE in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_ALLOC_FAILED)
        printf( "CUBLAS_STATUS_ALLOC_FAILED in %s at line %d\n", file, line );
       else if(error == CUBLAS_STATUS_NOT_INITIALIZED)
        printf( "CUBLAS_STATUS_NOT_INITIALIZED in %s at line %d\n", file, line );
       else
        printf( "Error code: %d in %s at line %d\n", error, file, line );
        exit( -1 );
    }
}

#define HANDLE_ERROR( error ) (handleError( (error) , __FILE__ , __LINE__ ))
#define HANDLE_ERROR_CUBLAS( error ) (handleError_cublas( (error) , __FILE__ , __LINE__ ))


#endif
