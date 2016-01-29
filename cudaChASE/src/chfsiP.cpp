#include "../include/chfsi.h"
#include "../include/lanczos.h"
#include "../include/filter.h"
#include "../include/cuda_util.h"
#include <cuda_runtime.h>
#include "../include/inst.h"

extern int MKL_NUM_THREADS;
extern int OMP_NUM_THREADS;

extern gpu_data_t gpuData; //TODO don't make this global
const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

void chfsi(MKL_Complex16* const H, int N, MKL_Complex16* V, 
      MKL_Complex16* W, double* ritzv, int nev, 
      const int nex, int deg, const double tol, 
      const int int_mode, const int int_opt, const int int_arch)
{
   cout << "ChFSI..." << endl;
   PUSH_RANGE("chsubsp",1);

   MKL_Complex16 *RESULT = W;
   MKL_Complex16 *swap = NULL; // Just a pointer for swapping V and W.
   MKL_Complex16 *tau = NULL; // Auxiliary vector needed for LAPACK functions.
   MKL_Complex16 *S = NULL;

   MKL_Complex16 alpha, beta; // Complex numbers needed for LAPACK functions.
   cuDoubleComplex alpha_d, beta_d;

   double *Lambda = NULL; // Pointer to the first element containing a not yet converged eigenvalue.
   double* resid = new double[nev]; // Residuals.
   double norm1, norm2; // Used in computations of the degrees and the residuals.

   int notneeded, INFO = 0; // Needed for LAPACK.

   //  omp_filtered = 0; //just in case - not really needed??

   int i, j; //iterators in smaller loops - used often
   int old; // Used for temporary storing a value.

   int blk = nev + nex; // Block size for the algorithm. CONSTANT. Used for LAPACK functions.
   int block = nev + nex; // Number of non converged eigenvalues. NOT constant!
   int filteredVectors = 0;
   int blk2, blk3; 

   const int iONE = 1; // Needed for LAPACK.

   // Variables needed for optimization.
   double c, e; // c ... centre of the segment. e ... half-length of the segment.
   int* iopt[5]    = {NULL};
   double* dopt[2] = {NULL};
   for(i = 0 ; i < 5 ; ++i) iopt[i] = new int[nev];
   for(i = 0 ; i < 2 ; ++i) dopt[i] = new double[nev];

   //------------------------COMPUTING-THE-OPTIMAL-WORKSPACES------------------------------- 
   double abstol = dlamch_( "S" );

   MKL_Complex16 * A = new MKL_Complex16[blk*blk]; // For LAPACK.
   MKL_Complex16 * X = new MKL_Complex16[blk*blk]; // For LAPACK.
   cuDoubleComplex *A_d, *X_d, *swap_d;
   size_t sizeA_d = 0, sizeX_d = 0;
   sizeA_d = blk*blk * sizeof(cuDoubleComplex);
   HANDLE_ERROR( cudaMalloc((void**) &(A_d), sizeA_d) );  //sync
   sizeX_d = blk*blk * sizeof(cuDoubleComplex);
   HANDLE_ERROR( cudaMalloc((void**) &(X_d), sizeX_d) );  //sync

   int *isuppz = new int[2*blk];

   // Calculate optimal memory size needed for the algorithm.
   int lzmem  =  0,  ldmem =  0,  limem =  0;
   int llzmem = -1, lldmem = -1, llimem = -1;

   MKL_Complex16 *zmem = new MKL_Complex16[1];
   double *dmem = new double[1]; 
   int *imem = new int[1];

   PUSH_RANGE("zheevr ",1);
   zheevr_("V", "A", "U", &blk, A, &N, NULL, NULL, NULL, NULL,
         &abstol, &notneeded, Lambda, X, &N, isuppz,
         zmem, &llzmem, dmem, &lldmem, imem, &llimem, &INFO);
   POP_RANGE

   magma_int_t magma_info;
   magma_int_t magma_err;
   int nb = magma_get_zgeqrf_nb( N );
   int zmem_size = ((2*min(N, blk) + ((blk+31)/32)*32 )*nb) * 10 * sizeof(cuDoubleComplex);
   cuDoubleComplex *zmem_d;
   HANDLE_ERROR( cudaMalloc((void**) &(zmem_d), zmem_size) );  //sync

   // Calculating the optimal sizes.
   PUSH_RANGE("zgeqrf ",2);
   if((int)zmem[0].real() > lzmem) lzmem = (int)zmem[0].real();
   if((int)dmem[0] > ldmem) ldmem = (int)dmem[0];
   if(imem[0] > limem) limem = imem[0]; llzmem = -1;  

   zgeqrf_(&N, &blk, W, &N, tau, zmem, &llzmem, &INFO);
   if((int)zmem[0].real() > lzmem) lzmem = (int)zmem[0].real(); llzmem = -1;
   zungqr_( &N, &blk, &blk, W, &N, tau, zmem, &llzmem, &INFO );
   if((int)zmem[0].real() > lzmem) lzmem = (int)zmem[0].real();
   POP_RANGE

   delete[] zmem; delete[] dmem; delete[] imem;

   // Allocation of necessary memory.
   zmem = new MKL_Complex16[lzmem];
   dmem = new double[ldmem];
   imem = new int[limem];
   //---------------------------------------------------------------------------------------

   //-----------------------GENERATE-A-RANDOM-VECTOR----------------------------------------
   // Create randomVector from Gaussian distribution with parameters 0,1. Needed for lanczos().
   MKL_Complex16 *randomVector = new MKL_Complex16[N];
   VSLStreamStatePtr stream;
   vslNewStream(&stream, VSL_BRNG_MT19937, rand()%1000+100);
   vdRngGaussian(VSL_METHOD_DGAUSSIAN_ICDF, stream, 2*N, (double*)randomVector, 0.0, 1.0);
   vslDeleteStream(&stream);
   //---------------------------------------------------------------------------------------

   int converged = 0; // Number of converged eigenpairs.
   int iteration = 0; // Current iteration.

   // Needed for the filter().
   int omp_num_threads;
   int omp_rank;
   int omp_start;
   int omp_work;

   // To store the approximations obtained from lanczos().
   double lowerb, upperb, lambda;

   omp_set_num_threads(1);
   mkl_set_num_threads(MKL_NUM_THREADS);

   omp_time[END_FILTER] = 0.0;
   omp_time[END_RR] = 0.0;
   omp_time[END_CONV] =0.0;

   omp_time[BGN_TOTAL] = omp_get_wtime(); //*********************************************TIME

   //-------------------------  HtD transfer -------------------------------
#ifdef MULTI_GPU
   cudaSetDevice(0);
   HANDLE_ERROR( cudaMemcpyAsync(gpuData.H_d, H, gpuData.sizeA, cudaMemcpyHostToDevice, gpuData.stream0)); //sync
   cudaSetDevice(1);
   HANDLE_ERROR( cudaMemcpyAsync(gpuData.H1_d, H + N * gpuData.work_per_gpu[0], N * gpuData.work_per_gpu[1] * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, gpuData.stream1)); //sync
   cudaSetDevice(0);
#else
   HANDLE_ERROR( cudaMemcpyAsync(gpuData.H_d, H, gpuData.sizeA, cudaMemcpyHostToDevice,gpuData.stream0)); //sync
#endif
   //-----------------------------------------------------------------------


   omp_time[BGN_LANCZOS] = omp_get_wtime(); //*******************************************TIME

   // OUTPUT: (APPROX) upperb; (RANDOM) ritzv, upperb.
   // ritzv  ... Vector - approximations of lambda_1,...,lambda_{nex+nev}.
   // upperb ... Upper bound for the interval.
   PUSH_RANGE("lanczos",3);
   lanczos(H, randomVector, N, blk, omp_lanczos, tol, int_mode, int_mode ? ritzv : NULL, &upperb);  
   POP_RANGE

   omp_time[END_LANCZOS] = omp_get_wtime() - omp_time[BGN_LANCZOS]; //*******************TIME
   cout << "TIME - lanczos: " << omp_time[END_LANCZOS] << endl;

   delete[] randomVector; // Not needed anymore.


   while(converged < nev && iteration < omp_maxiter)
   {
      cout << "ChFSI: " << iteration << ". iteration." << endl;

      Lambda = ritzv + converged; // Pointer to the first non converged eigenvalue.

      lambda = minValue(ritzv, blk); // Approximation for the smallest eigenvalue. (nev + nex ??)
      lowerb = maxValue(Lambda, block); // Lower bound for the interval.

      omp_set_num_threads(OMP_NUM_THREADS); 
      mkl_set_num_threads(OMP_NUM_THREADS);

      if( int_opt == OMP_NO_OPT || iteration == 0 )
      {
         filteredVectors += block*deg; // Number of filtered vectors.

         omp_time[BGN_FILTER] = omp_get_wtime(); //*************************************TIME

         //-------------------------------------FILTER--------------------------------------
         cout << "FILTER..." << endl;
         PUSH_RANGE("filter",4);
         filter(H, V + N*converged, N, block, deg, lambda, lowerb, upperb, W + N*converged, &gpuData);
         POP_RANGE

         if(deg%2 == 0) // To 'fix' the situation if the filter swapped them.
         {
            swap = V; V = W; W = swap;
         }
         cout << "FILTER done." << endl;
         //----------------------------------------------------------------------------------
         // Initialization of the permutation. (After the zheevr_ the eigenpairs will be sorted, so it's the identity.)
         if( int_opt == OMP_OPT )
            for(i = 0 ; i < nev ; ++i)  iopt[PERM][i] = i;

         omp_time[END_FILTER] += (omp_get_wtime() - omp_time[BGN_FILTER]); //************TIME

         cout << "TIME - filter: " << omp_time[END_FILTER] << endl;

      }
      else
      {
         PUSH_RANGE("filter prepare ",5);
         c = (upperb + lowerb) / 2; // Centre of the interval.
         e = (upperb - lowerb) / 2; // Half-length of the interval.

         alpha = MKL_Complex16 (1.0, 0.0);
         beta  = MKL_Complex16 (0.0, 0.0);

         nev -= converged;
         ZGEMV("C", &N, &nev, &alpha, V+N*converged, &N, V, &iONE, &beta, W+N*converged+converged, &iONE);
         nev += converged;

         //Find the minimal degrees, and a maximal one.
         deg = omp_degmax;
         dopt[RHO][0] = (ritzv[0]-c)/e;
         norm1 = sqrt(dopt[RHO][0]*dopt[RHO][0]-1);
         norm2 = fabs(dopt[RHO][0] + norm1);
         norm1 = fabs(dopt[RHO][0] - norm1);
         dopt[RHO][0] = (norm1 > norm2) ? norm1 : norm2;
         for(i = converged ; i < nev ; ++i)
         {
            dopt[RHO][i] = (ritzv[i]-c)/e;
            norm1 = sqrt(dopt[RHO][i]*dopt[RHO][i]-1);
            norm2 = fabs(dopt[RHO][i] + norm1);
            norm1 = fabs(dopt[RHO][i] - norm1);
            dopt[RHO][i] = (norm1 > norm2) ? norm1 : norm2;

            dopt[SP][i] = abs(W[N*converged+i]);

            iopt[MINDEG][i] = ceil(fabs( log(resid[i]/tol)/log(dopt[RHO][i]) ));

            norm1  = log(dopt[RHO][0]);
            norm2  = log(fabs( dopt[RHO][i]/log(dopt[RHO][0]/dopt[RHO][i]) ));
            norm2 += log(resid[i]/dopt[SP][i]);
            iopt[MAXDEG][i] = floor(norm2/norm1);
            if( iopt[MAXDEG][i] > 0 && deg > iopt[MAXDEG][i])
               deg = iopt[MAXDEG][i];
         }

         // No minimal degree should be greater than the smallest maximal degree.
         // To be sure the eigenpairs will converge, the (minimal) degree used is augmented a bit (delta).
         for(i = converged ; i < nev ; ++i)
            if(iopt[MINDEG][i] + omp_delta >= deg)
               iopt[MINDEG][i] = deg;
            else 
               iopt[MINDEG][i] += omp_delta;

         // Sort the minimal degrees used within the filter.
         memcpy(iopt[SORTED]+converged, iopt[MINDEG]+converged, (nev-converged)*sizeof(int));
         qsort(iopt[SORTED]+converged, nev-converged, sizeof(int), sortComp);
         if(iopt[MINDEG][nev-1] < deg && iopt[MINDEG][nev-1] > 1)
            deg = iopt[MINDEG][nev-1];

         // Find the local permutation, for the not yet converged vectors.
         // 1. Fix those which do not need to be moved.
         for(i = converged ; i < nev ; ++i)
            if(iopt[SORTED][i] == iopt[MINDEG][i])
            {
               iopt[WHERE][i] = i;
               iopt[ALLOWED][i] = 0;
            }
            else 
               iopt[ALLOWED][i] = 1;
         // 2. Look how to move the rest.
         for(i = converged ; i < nev ; ++i) 
            if(iopt[SORTED][i] != iopt[MINDEG][i])
            {
               // THIS HAS TO BE A BINARY SEARCH !!!
               for(j = converged ; j < nev ; ++j)
                  if(iopt[SORTED][j] == iopt[MINDEG][i] && iopt[ALLOWED][j])
                     break;
               iopt[WHERE][i] = j;
               iopt[ALLOWED][j] = 0;
            }
         // 3. Find the inverse permutation. (For the part needed, not for the already converged ones.)
         for(i = converged ; i < nev ; ++i)
         {
            iopt[PINV][iopt[WHERE][i]] = i;
            iopt[ALLOWED][i] = 0;
         }

         // Permute vectors to be able to run the filter easily.
         applyPerm2vect(iopt, nev, converged, W+N*converged, N, V);

         for(i = converged ; i < nev ; ++i)
            filteredVectors += iopt[SORTED][i];
         filteredVectors += nex*deg;	  
         POP_RANGE

         omp_time[BGN_FILTER] = omp_get_wtime(); //******************************************TIME

         //----------------------------------------FILTER----------------------------------------
         cout << "FILTER MODIFIED..." << endl;
         PUSH_RANGE("filter modified",6);
         filterModified(H, V + N*converged, N, block, nev, deg, iopt[SORTED]+converged,
               lambda, lowerb, upperb, W + N*converged, block-nex, &gpuData);
         POP_RANGE

         if(deg%2 == 0) // To 'fix' the situation if the filter swapped them.
         {
            swap = V; V = W; W = swap;
         }
         cout << "FILTER MODIFIED done." << endl;
         //--------------------------------------------------------------------------------------
         // Sort the permutation. (After the zheevr_ the eigenpairs will be sorted.)
         qsort(iopt[PERM]+converged, nev-converged, sizeof(int), sortComp);

         omp_time[END_FILTER] += (omp_get_wtime() - omp_time[BGN_FILTER]); //****************TIME

         cout << "TIME - filter: " << omp_time[END_FILTER] << endl;

      }
      omp_set_num_threads(1); 
      mkl_set_num_threads(MKL_NUM_THREADS);   

      omp_time[BGN_RR] = omp_get_wtime(); //**************************************************TIME

      //-------------------------------------RAYLEIGH-RITZ----------------------------------------
      // [W, ~ ] = qr(W);
      tau = V + N*converged;
      PUSH_RANGE("zgeqrf",7);
//      ZGEQRF(&N, &blk, W, &N, tau, zmem, &lzmem, &INFO);   
      HANDLE_ERROR( cudaMemcpy(gpuData.W_d, W, gpuData.sizeW, cudaMemcpyHostToDevice)); //sync
      magma_err = magma_zgeqrf_gpu( N, blk, gpuData.W_d, N, (cuDoubleComplex*)tau, zmem_d, &magma_info);
      POP_RANGE

      PUSH_RANGE("zungqr",8);
//      ZUNGQR(&N, &blk, &blk, W, &N, tau, zmem, &lzmem, &INFO);
      magma_err = magma_zungqr_gpu( N, blk, blk, gpuData.W_d, N, (cuDoubleComplex*)tau, zmem_d, nb, &magma_info);
      POP_RANGE

      //.................................... G = W' * H * W;      
      alpha = MKL_Complex16 (1.0, 0.0);
      beta  = MKL_Complex16 (0.0, 0.0);      
      alpha_d.x = 1.0; alpha_d.y = 0.0;
      beta_d.x = 0.0; beta_d.y = 0.0;

      blk2 = block*block;

      omp_set_num_threads(OMP_NUM_THREADS);
      mkl_set_num_threads(OMP_NUM_THREADS);

      //          H * W  
      PUSH_RANGE("GEMM 1",9);
//      ZGEMM("N", "N", &N, &block, &N, &alpha, (const MKL_Complex16*) (H), &N, (const MKL_Complex16*)(W + N*converged), &N, &beta, V + N*converged , &N);

//      HANDLE_ERROR( cudaMemcpy(gpuData.W_d, W, gpuData.sizeW, cudaMemcpyHostToDevice)); //sync
      HANDLE_ERROR( cudaMemcpy(gpuData.V_d, V, gpuData.sizeV, cudaMemcpyHostToDevice)); // necessary because tau points to part of V
      HANDLE_ERROR_CUBLAS( cublasZgemm( gpuData.handle, //updates gpuData.V
              CUBLAS_OP_N, CUBLAS_OP_N,
              N, block, N, &alpha_d,
              gpuData.H_d, N, gpuData.W_d + N*converged, N, &beta_d, gpuData.V_d + N*converged, N) );
      POP_RANGE

      //***  W' *(     )
      for(i=0; i<blk2; ++i) A[i] = 0;

      S= new MKL_Complex16[blk2];

      PUSH_RANGE("GEMM 2",9);
//      ZGEMM("C", "N", &block, &block, &N, &alpha,(const MKL_Complex16*)(W + N*converged ), &N, (const MKL_Complex16*)(V + N*converged ), &N, &beta, S, &block);
      HANDLE_ERROR_CUBLAS( cublasZgemm( gpuData.handle, //creates A_d
              CUBLAS_OP_C, CUBLAS_OP_N,
              block, block, N, &alpha_d,
              gpuData.W_d + N*converged, N, gpuData.V_d + N*converged, N, &beta_d, A_d, block) );
      HANDLE_ERROR( cudaMemcpy(S, A_d, blk2*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) ); //TODO not required once zheevr in on gpu
      POP_RANGE

      ZAXPY(&blk2, &alpha, S, &iONE, A, &iONE);

      delete[] S;
      //.....................................................

      omp_set_num_threads(1);
      mkl_set_num_threads(MKL_NUM_THREADS);      

      PUSH_RANGE("ZHEEVR",11);
      ZHEEVR("V", "A", "U", &block, A, &block, 
            NULL, NULL, NULL, NULL, &abstol, &notneeded, Lambda, 
            X, &blk, isuppz, zmem, &lzmem, dmem, &ldmem, imem, &limem, &INFO);
      POP_RANGE

      blk3 = block - nex;

      mkl_set_num_threads(OMP_NUM_THREADS);
      omp_set_num_threads(OMP_NUM_THREADS);

      // W = W*Q;
      PUSH_RANGE("GEMM 3",9);
//      ZGEMM("N", "N", &N, &block, &block, &alpha,
//            (const MKL_Complex16*)(W + N*converged), &N, (const MKL_Complex16*)X, &blk,
//            &beta, V + N*converged, &N );      
      HANDLE_ERROR( cudaMemcpy(X_d, X, sizeX_d, cudaMemcpyHostToDevice)); //sync
      HANDLE_ERROR_CUBLAS( cublasZgemm( gpuData.handle, //updates gpuData.V + N*converged
              CUBLAS_OP_N, CUBLAS_OP_N,
              N, block, block, &alpha_d,
              gpuData.W_d + N*converged, N, X_d, blk, &beta_d, gpuData.V_d + N*converged, N) );
      POP_RANGE
      //------------------------------------------------------------------------------------------ 

      omp_time[END_RR] += (omp_get_wtime() - omp_time[BGN_RR]); //****************************TIME
      cout << "TIME - RR: " << omp_time[END_RR] << endl;

      omp_time[BGN_CONV] = omp_get_wtime(); //************************************************TIME

      //-----------------------------COMPUTING-THE-RESIDUALS--------------------------------------   
      // The extra vectors do not need to be checked for convergence.

      //block -=nex; // Instead used: blk3 = block - nex.
      PUSH_RANGE("GEMM 4",9);
//      ZGEMM("N", "N", &N, &blk3, &N, &alpha, (const MKL_Complex16*) (H), &N, 
//            (const MKL_Complex16*)(V + N*converged), &N, &beta, W + N*converged, &N );
      HANDLE_ERROR_CUBLAS( cublasZgemm( gpuData.handle, //updates gpuData.V + N*converged
              CUBLAS_OP_N, CUBLAS_OP_N,
              N, block, N, &alpha_d,
              gpuData.H_d, N, gpuData.V_d + N*converged, N, &beta_d, gpuData.W_d + N*converged, N) );
      HANDLE_ERROR( cudaMemcpy(V, gpuData.V_d, gpuData.sizeV, cudaMemcpyDeviceToHost) ); //prob not req. TODO
      HANDLE_ERROR( cudaMemcpy(W, gpuData.W_d, gpuData.sizeW, cudaMemcpyDeviceToHost) ); //prob not req.
      POP_RANGE
      //block += nex;

      for( i = converged ; i < nev ; ++i )
      {
         beta  = MKL_Complex16 (-ritzv[i], 0.0);
         ZAXPY(&N, &beta, V + N*i, &iONE, W + N*i, &iONE);

         norm1 = DZNRM2(&N, (const MKL_Complex16*)W + N*i, &iONE);
         norm2 = DZNRM2(&N, (const MKL_Complex16*)V + N*i, &iONE);

         resid[i] = norm1/norm2;
      }
      //-------------------------------------------------------------------------------------------

      //------------------------CHECKING-WHICH-EIGENPAIRS-CONVERGED--------------------------------
      old = converged;
      for(i = converged ; i < nev ; ++i)
      {
         if(resid[i] < tol) // If i-th eigenpair converged.
         { // Lock the eigenpair.
            swapEigPair(i, converged, W+N*converged, N, V, ritzv, iopt[PERM], int_opt); 
            // iopt[PERM] will be initialized and unimportant if int_opt==OMP_NO_OPT.
            ++converged; // Update the number of converged ones.
         }
      }
      block -= (converged-old); // Number of not yet converged.
      cout << "block... " << block << endl;
      //-------------------------------------------------------------------------------------------

      iteration += 1;
      memcpy(W + N*old, V + N*old, N*(converged-old)*sizeof(MKL_Complex16)); // So W is the solution.??

      omp_time[END_CONV] += (omp_get_wtime() - omp_time[BGN_CONV]); //*************************TIME
      cout << "TIME - conv: " << omp_time[END_CONV] << endl;

   } // while ( converged < nev && iteration < omp_maxiter )

   //---------------------------SORT-EIGENPAIRS-ACCORDING-TO-EIGENVALUES----------------------------
   if ( int_opt == OMP_OPT )
   {
      // 1. Find the inverse permutation.
      for(i = 0 ; i < nev ; ++i)
      {
         iopt[PINV][iopt[PERM][i]] = i;
         iopt[ALLOWED][i] = 0;
      }    
      // 2. Rearrange the eigenpairs.
      applyPerm2eigPair(iopt, nev, W, N, V, ritzv);
   }
   //------------------------------------------------------------------------------------------------

   omp_time[END_TOTAL] = omp_get_wtime() - omp_time[BGN_TOTAL]; //*****************************TIME

   omp_iteration = iteration;

   if( V != RESULT )
   {
      if( W != RESULT ) {fprintf(stderr, "Something went terribly wrong!\n"); exit(EXIT_FAILURE);}
      else memcpy(W , V, N*(converged)*sizeof(MKL_Complex16));
   }

   for(i = 0 ; i < 5 ; ++i) delete[] iopt[i];
   for(i = 0 ; i < 2 ; ++i) delete[] dopt[i];

   delete[] zmem; delete[] dmem; delete[] imem;
   delete[] isuppz; delete[] resid;
   delete[] A; delete[] X;
   cudaFree (A_d); 
   cudaFree (X_d); 
   cudaFree (zmem_d); 

   cout << "ChFSI done." << endl;
   POP_RANGE
   return;
}

// Find the minimal value stored in vector v of dimension N.
double minValue( double *v, int N )
{
   if( v == NULL ) exit( EXIT_FAILURE );
   int i;
   double ret = v[0];
   for( i = 1 ; i < N ; ++i )
      if( v[i] < ret ) ret = v[i];
   return ret;
}

// Find the maximal value stored in vector v of dimension N.
double maxValue( double *v, int N )
{
   if( v == NULL ) exit( EXIT_FAILURE );
   int i;
   double ret = v[0];
   for( i = 1 ; i < N ; ++i )
      if( v[i] > ret ) ret = v[i];
   return ret;
}

// Function used for qsort.
int sortComp (const void * a, const void * b)
{
   if ( *(int*)a <  *(int*)b ) return -1;
   if ( *(int*)a == *(int*)b ) return 0;
   if ( *(int*)a >  *(int*)b ) return 1;
   cout << "Wrong parameters to sortComp" << endl;
   return -2;
}

// Apply permutation P to nonconverged vectors (stored in V, determined by shift).
void applyPerm2vect(int** perm, int n, int shift, MKL_Complex16* tmp, 
      int N, MKL_Complex16* V)
{
   int k, K;
   int start;
   for(k = shift ; k < n ; ++k)
      if(perm[PINV][k] != k && perm[ALLOWED][k] == 0)
      {
         start = k;
         K = perm[PINV][k];
         memcpy(tmp, V+N*K, N*sizeof(MKL_Complex16));
         while(K != start)
         {         
            memcpy(V+N*K, V+N*perm[PINV][K], N*sizeof(MKL_Complex16));
            perm[ALLOWED][K] = 1;
            K = perm[PINV][K];
         }
         memcpy(V+N*start, tmp, N*sizeof(MKL_Complex16));      
      }
}

// Swap i-th and j-th eigenpair and update the permutation.
void swapEigPair(int i, int j, MKL_Complex16* ztmp,
      int N, MKL_Complex16* V, double *Lambda, int* P, int int_opt)
{
   int itmp;
   double dtmp;
   if(i == j) return;

   if ( int_opt == OMP_OPT )
   {
      itmp = P[i]; P[i] = P[j]; P[j] = itmp;
   }
   dtmp = Lambda[i]; Lambda[i] = Lambda[j]; Lambda[j] = dtmp;

   memcpy(ztmp,  V+N*i, N*sizeof(MKL_Complex16));
   memcpy(V+N*i, V+N*j, N*sizeof(MKL_Complex16));
   memcpy(V+N*j, ztmp,  N*sizeof(MKL_Complex16));

   return;
}

// Apply permutation to eigenvectors (in V) and eigenvalues (in Lambda)
void applyPerm2eigPair(int** perm, int n, MKL_Complex16* ztmp,
      int N, MKL_Complex16* V, double *Lambda)
{
   int k, K;
   int start;  
   int itmp;
   double dtmp;

   for(k = 0 ; k < n ; ++k)
      if(perm[PINV][k] != k && perm[ALLOWED][k] == 0)
      {
         start = k;
         K = perm[PINV][k];
         memcpy(ztmp, V+N*K, N*sizeof(MKL_Complex16));
         dtmp = Lambda[K];
         //itmp = P[K];
         while(K != start)
         {         
            memcpy(V+N*K, V+N*perm[PINV][K], N*sizeof(MKL_Complex16));
            Lambda[K] = Lambda[perm[PINV][K]];
            //P[k] = P[perm[PINV][K]];
            perm[ALLOWED][K] = 1;
            K = perm[PINV][K];
         }
         memcpy(V+N*start, ztmp, N*sizeof(MKL_Complex16));     
         Lambda[start] = dtmp;
         //P[start] = itmp;
      }
}

/*void get_filtered(int* filtered)
  {
 *filtered = omp_filtered;
 return;
 }*/

void get_iteration(int* iteration)
{
   *iteration = omp_iteration;
   return;
}


void get_time(double* time)
{
   for (int i = 0; i < 5; ++i)
      time[i] = omp_time[5+i];
   return;
}
