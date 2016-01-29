#include "chfsi.h"
#include "filter.h"

// Auxiliary function. Writes a matrix A of dimensions m x n to stdout.
template<typename T> void ispisi(T *A, int m, int n)
{
   int k=0;
   cout << endl;
   for(int i=0; i<m; i++)
   {
      for(int j=0; j<n; j++)
      {
         cout << A[k] << " ";
         k++;
      }
      cout << endl;
   }
   cout << endl;
   return;
}
/*
   Reads/Writes a matrix/vector of type T from/to a file.
   A     ... Matrix/Vector that needs to be read/written from/to file. OUTPUT.
   path  ... Path to the file from/in which to read/write.
   app   ... ".vct" or ".vls"
   index ... IterationIndex of the problem. Specifies the file.
   size  ... Number of elements of type T to be read/written.
   opt   ... 'R' or 'W'. Specifies if the mode is Read or Write. (Case insensitive.)
   */
template<typename T> void myreadwrite(T* A, const char* path, const char* app, int index, int size, const char opt)
{
   char mat[200];
   char str[200];
   FILE *stream=NULL;

   mat[0] = '\0';
   strcat( mat, path);
   sprintf( str, "gmat  1 %2d", index );
   strcat( mat, str );
   strcat( mat, app);

   switch(opt)
   {
      case 'r': case 'R': // READ
         stream = fopen( mat, "rb" );
         if(stream == NULL){
            std::cout<< "FILE: "<<mat<< " cannot be opened!\n";
            exit(EXIT_FAILURE);
         }

         if ( fread( A, sizeof(T), size, stream ) != size )           
            exit( EXIT_FAILURE );
         fclose( stream );

         cout << "Reading from " << mat << "." << endl; // ??

         break;
      case 'w': case 'W': // WRITE
         stream = fopen( mat, "wb" );
         if ( fwrite( A, sizeof(T), size, stream ) != size )
            exit( EXIT_FAILURE );
         fclose( stream );

         cout << "Writing to " << mat << "." << endl; // ??

         break;
      default:
         cout << "Wrong read/write argument." << endl;
   }
   return;
}

//int MKL_NUM_THREADS;
//int OMP_NUM_THREADS;

int main(int argc, char* argv[])
{

   mkl_set_dynamic(0);
   omp_set_dynamic(0); 
//   MKL_NUM_THREADS = mkl_get_max_threads(); 
//   OMP_NUM_THREADS = omp_get_max_threads();

   if(argc != 15)
   {
      cout << "N nev nex deg bgn end tol mat mode opt arch eigp res info ZHEEVR" << endl;
      return EXIT_FAILURE;
   }

   int N   = atoi(argv[1]); // Size of the system.
   int nev = atoi(argv[2]); // Number of wanted eigenvalues.
   int nex = atoi(argv[3]); // Number of additional eigenvalues (see ../USAGE).
   int deg = atoi(argv[4]); // Degree for the filter.
   int bgn = atoi(argv[5]); // First iteration index in the sequence of matrices (inclusive).
   int end = atoi(argv[6]); // Last iteration index in the sequence of matrices (inclusive).
   double tol = atof(argv[7]); // Tolerance for the algorithm.

   string path_in = argv[8]; // Path to the location of the matrices (see ../USAGE).
   string mode = argv[9]; // Random/Approximate.
   string opt  = argv[10]; // Optimization/No optimization.
   string arch = argv[11]; // CPU/GPU/Xeon Phi.
   string path_eigp = argv[12]; // '_' or the path to approx. eigenpairs (see ../USAGE).
   string path_out = argv[13]; // Path to the location where the solutions will be saved.
   string path_name = argv[14]; // Name of the output file (see USAGE).

   cout << setw(8) << "N" << setw(60) << N << endl
      << setw(8) << "nev" << setw(60) << nev << endl
      << setw(8) << "nex" << setw(60) << nex << endl
      << setw(8) << "deg" << setw(60) << deg << endl
      << setw(8) << "bgn" << setw(60) << bgn << endl
      << setw(8) << "end" << setw(60) << end << endl
      << setw(8) << "tol" << setw(60) << tol << endl
      << setw(8) << "mat" << setw(60) << path_in << endl
      << setw(8) << "mode" << setw(60) << mode << endl
      << setw(8) << "opt" << setw(60) << opt << endl
      << setw(8) << "arch" << setw(60) << arch << endl
      << setw(8) << "eigp" << setw(60) << path_eigp << endl
      << setw(8) << "res" << setw(60) << path_out << endl
      << setw(8) << "info" << setw(60) << path_name << endl;

   int blk = nev + nex; // Block size for the algorithm.

   MKL_Complex16 *H = (MKL_Complex16*) _mm_malloc(sizeof(MKL_Complex16)*N*N,64); // Matrix representing the generalized eigenvalue problem.
   MKL_Complex16 *V = (MKL_Complex16*) _mm_malloc(sizeof(MKL_Complex16)*N*blk,64); // Matrix which stores the approximate eigenvectors (INPUT to chfsi).
   MKL_Complex16 *W = (MKL_Complex16*) _mm_malloc(sizeof(MKL_Complex16)*N*blk,64); // Matrix which stores the eigenvectors (OUTPUT of the chfsi function).

   const int matrix_elementsA = N*N;
   const int matrix_elementsW = N*blk;
#pragma offload_transfer target(mic:0) nocopy(H:length(matrix_elementsA) ALLOC) 
#pragma offload_transfer target(mic:0) nocopy(V:length(matrix_elementsW) ALLOC) 
#pragma offload_transfer target(mic:0) nocopy(W:length(matrix_elementsW) ALLOC)
#ifdef MULTIPLE_MICS
#pragma offload_transfer target(mic:1) nocopy(H:length(matrix_elementsA) ALLOC) 
#pragma offload_transfer target(mic:1) nocopy(V:length(matrix_elementsW) ALLOC) 
#pragma offload_transfer target(mic:1) nocopy(W:length(matrix_elementsW) ALLOC)
#endif

   double * Lambda = new double[blk];

   int int_mode, int_opt, int_arch; // Control variables.
   int INFO; // Used for zheevr_ and ZHEEVR.
   int filtered; // Number of filtered vectors.
   int iterations; // Number of iterations.

   string app; // Used for myreadwrite().

   //------------------CALCULATE-THE-OPTIMAL-SIZES-OF-WORKSPACES------------------

   MKL_Complex16 *zmem = new MKL_Complex16[1];
   double *dmem = new double[1];
   int *imem = new int[1];
   int *isuppz = new int[2*blk];

   double vl, vu;
   int lzmem = -1, ldmem = -1, limem = -1;
   int il = 1, iu = blk,  notneeded;

   zheevr_("V", "I", "L", &N, H, &N, &vl, &vu, &il, &iu, &tol, &notneeded, Lambda, W, &N, isuppz, zmem, &lzmem, dmem, &ldmem, imem, &limem, &INFO);

   // Storing the optimal sizes.
   lzmem = (int)zmem[0].real; delete[] zmem;
   ldmem = (int)dmem[0]; delete[] dmem;
   limem = (int)imem[0]; delete[] imem;

   // Allocating the necessary memory.
   zmem = new MKL_Complex16[lzmem * sizeof(MKL_Complex16)];
   dmem = new double[ldmem * sizeof(double)];
   imem = new int[limem * sizeof(int)];

   //------------------------------------------------------------------------------

   //-----------------------------SET-CONTROL-VARIABLES----------------------------
   // Set the control variable for the mode: Approximate/Random.
   switch (mode[0])
   {
      case 'a':case 'A':
         int_mode = OMP_APPROX; // APPROX
         break;
      case 'r':case 'R':
         int_mode = OMP_RANDOM; // RANDOM
         break;
      default:      
         cout << "Wrong mode argument." << endl;
   }
   // Set the control variable for Optimization/No optimization.
   switch (opt[0])
   {
      case 'n':case 'N':
         int_opt = OMP_NO_OPT; // NO OPTIMIZATION
         break;
      case 'o':case 'O':
         int_opt = OMP_OPT; // OPTIMIZATION
         break; 
      default:     
         cout << "Wrong opt argument." << endl;
   }
   // Set the control variable that determines the architecture.
   switch (arch[0])
   {
      case 'c':case 'C':
         int_arch = OMP_CPU; // CPU
         break;
      case 'g':case 'G':
         int_arch = OMP_GPU; // GPU
         break; 
      case 'x':case 'X':
         int_arch = OMP_XEON_PHI; // XEON PHI
         break;
      default:     
         cout << "Wrong arch argument." << endl;
   }
   //-----------------------------------------------------------------------------

   double tt[5]={0.0};

   // Iterate over problems that need to be solved.
   for (int i = bgn; i <= end ; ++i)
   {      
//      mkl_set_num_threads(MKL_NUM_THREADS);
//      omp_set_num_threads(1);

      if (path_eigp == "_" && int_mode == OMP_APPROX && i == bgn )
      { // APPROX. No approximate pairs given.
         //---------------------------SOLVE-PREVIOUS-PROBLEM---------------------------
         app = ".bin"; // Read the matrix of the previous problem.
         myreadwrite<MKL_Complex16>(H, path_in.c_str(), app.c_str(), i-1, N*N, 'r');

         cout << "ZHEEVR..." << endl;

         // Solve the previous problem, store the eigenpairs in V and Lambda.
         ZHEEVR("V", "I", "L", &N, H, &N, &vl, &vu, &il, &iu, &tol, &notneeded, Lambda, V, &N, isuppz, zmem, &lzmem, dmem, &ldmem, imem, &limem, &INFO);

         cout << "ZHEEVR done." << endl;
         //----------------------------------------------------------------------------
         // In next iteration the solutions to this one will be used as approximations.
         path_eigp = path_out;
      }
      else if (int_mode == OMP_APPROX)
      { // APPROX. Approximate eigenpairs given.
         //-----------------------READ-APPROXIMATE-EIGENPAIRS---------------------------
         app = ".vct"; // Read approximate eigenvectors.
         myreadwrite<MKL_Complex16>(V, path_eigp.c_str(), app.c_str(), i-1, N*blk, 'r');

         app = ".vls"; // Read approximate eigenvalues.
         myreadwrite<double>(Lambda, path_eigp.c_str(), app.c_str(), i-1, blk, 'r');
         //-----------------------------------------------------------------------------
      }
      else
      { // RANDOM.
         // Randomize V.
         VSLStreamStatePtr randomStream;
         vslNewStream(&randomStream, VSL_BRNG_MT19937, 677);
//         vdRngGaussian(VSL_METHOD_DGAUSSIAN_ICDF, randomStream, 2*N*blk, (double*)V, 0.0, 1.0);
         vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, randomStream, 2*N*blk, (double*)V, 0.0, 1.0); //fix for the new icpc compiler 15.x
         vslDeleteStream(&randomStream);

         // Set Lambda to zeros. ( Lambda = zeros(N,1) )
         for(int j=0; j<blk; j++) Lambda[j]=0.0;
      }

      //------------------------------SOLVE-CURRENT-PROBLEM-------------------------------
      app = ".bin"; // Read the Hermitian matrix H from file.
      myreadwrite<MKL_Complex16>(H, path_in.c_str(), app.c_str(), i, N*N, 'r');

      //      chfsi(H, N, V, W, Lambda, nev, nex, deg, tol, int_mode, int_opt, int_arch, i);
      chfsi(H, N, V, W, Lambda, nev, nex, deg, tol, int_mode, int_opt, int_arch);

      //----------------------------------------------------------------------------------

      //-----------------------------STORE-THE-EIGENPAIRS---------------------------------
      app = ".vct"; // Write the eigenvectors to file.
      myreadwrite<MKL_Complex16>(W, path_out.c_str(), app.c_str(), i, N*blk, 'w');

      app = ".vls"; // Write the eigenvalues to file.
      myreadwrite<double>(Lambda, path_out.c_str(), app.c_str(), i, blk, 'w');
      //---------------------------------------------------------------------------------

      get_iteration(&iterations); // Get the number of iterations.

      double time[5];
      get_time(time); // Get the times needed for computations.

      tt[0] += time[0];
      tt[1] += time[1];
      tt[2] += time[2];
      tt[3] += time[3];
      tt[4] += time[4];

      //------------------------------PRINT-STUFF-TO-STDOUT---------------------------------??

      cout << endl
         << setw(15) << "i:" << setw(10) << i << "; " << endl
         << setw(15) << "iter:" << setw(10)  << iterations << "; " << endl
         << setw(15) << "TIME:" << setw(10) << time[0] << "; "
         << setw(8) << "lanczos:" << setw(10) << time[1]  << "; "
         << setw(8) << "filter:" << setw(10) << time[2]  << "; "
         << setw(8) << "RR:" << setw(10) << time[3]  << "; "
         << setw(8) << "conv:" << setw(10) << time[4] << "; "
         << endl << endl;

      //-----------------------------------------END-----------------------------------------

      //-------------------------------PRINT-INFO-TO-FILE------------------------------------
      char str[200];
      sprintf(str, "%s_%d_%d.xml", path_name.c_str(), int_mode, i);
      FILE *f = fopen(str,"w");

      fprintf(f,"INPUT\n\n");

      fprintf(f,  "N %d\n",N );
      fprintf(f,  "nev %d\n" ,nev );
      fprintf(f,  "nex %d\n" ,nex );
      fprintf(f,  "deg %d\n" ,deg );
      fprintf(f,  "bgn %d\n" ,bgn );
      fprintf(f,  "end %d\n" ,end );
      fprintf(f,  "tol %f\n" ,tol );
      fprintf(f,  "mat %s\n" ,&path_in[0] );
      fprintf(f,  "mode %s\n" ,&mode[0] );
      fprintf(f,  "opt %s\n" ,&opt[0] );
      fprintf(f,  "arch %s\n" ,&arch[0] );
      fprintf(f,  "eigp %s\n" ,&path_eigp[0] );
      fprintf(f,  "res %s\n\n" ,&path_out[0] );

      fprintf(f, "INFO \n\n" );
      fprintf(f,  "i: %d\n",i);
      fprintf(f,  "iterations: %d\n", iterations);

      fprintf(f, "TIME\n" ); 
      fprintf(f,  "total %f\n",time[0] );
      fprintf(f,  "lanczos: %f\n",  time[1] );
      fprintf(f,  "filter: %f\n",  time[2]  );
      fprintf(f,  "RR: %f\n", time[3]);
      fprintf(f,  "conv: %f\n",  time[4]);
      fprintf(f, "\n");
      fclose(f);

      //--------------------------------------------------------------------------------------

   } // for(int i = bgn; i <= end; ++i)

   cout << "---------------TIME---------------" << endl << endl;

   cout << "ChFSI: " << endl
      << setw(16) << "total" << setw(50) << tt[0] << endl
      << setw(16) << "lanczos:" << setw(50) << tt[1] << endl
      << setw(16) << "filter:" << setw(50) << tt[2]  << endl
      << setw(16) << "RR:" << setw(50) << tt[3]  <<  endl
      << setw(16) << "conv:" << setw(50) << tt[4]
      << endl;


   _mm_free(H); _mm_free(V); _mm_free(W);
   delete[] Lambda;
   delete[] zmem; delete[] dmem; delete[] imem; 
   delete[] isuppz;

   return 0;
}
