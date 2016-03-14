#ifndef CHASE_CUDA_UTIL_H
#define CHASE_CUDA_UTIL_H

#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cublas_v2.h"
#include <omp.h>
#include <stdio.h>
#include <magma.h>

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

#endif // CHASE_CUDA_UTIL_H
