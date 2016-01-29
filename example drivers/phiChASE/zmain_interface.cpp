//#include "elemental.hpp"
using namespace std;
//using namespace elem;
#include <string>
#include <iostream>
#include <sstream> // std::ostringstream
#include <fstream>
#include <iomanip> // std::setfill, std::setw, std::setprecision
#include <cstdio>
#include <mkl.h>
#include <time.h>
typedef double R;
typedef MKL_Complex16 C;

#include "phichase_fleur.h"
//#include "../../../tools/IO/io.hpp"




void read_matrix(C *A, int N, char* filename){
  
  FILE* stream = fopen(filename, "rb");
  if (stream == NULL)
  {
	  cerr << "Couldn't open file " << string(filename) << endl;
	//exit(-1);
  }
  
  int row;
  double re,im;
  for (int r = 0; r < N; ++r) {
	  row=r*N;
	  for (int c = 0; c < N; ++c) {
			//stream >> *(A_glob + N*c + r);
			fread(reinterpret_cast<void*>(&re), sizeof(double), 1, stream);
			fread(reinterpret_cast<void*>(&im), sizeof(double), 1, stream);
			A[row+c].real=re;
			A[row+c].imag=im;
			
	  }

  }	
}
 
int main(int argc, char* argv[])
{
  //Initialize(argc, argv);
 
  const int N   = 256; //Input<int>("--n",   "problem size");
  const int nev = 54; //Input<int>("--nev", "number of wanted eigenpairs");
  const int nex = 6; //Input<int>("--nex", "extra block size", 40);
  const int deg = 3; //Input<int>("--deg", "degree", 25);
  const int bgn = 0; //Input<int>("--bgn", "ell, begin");           
  const int end = 53; //Input<int>("--end", "ell, end");
  const R   tol = 1e-10; //Input<R>  ("--tol", "tolerance", 1e-10);

  const int algbs  = 128; //Input("--algbs",  "algorithmic block size", 128);
  const int height = 1; //Input("--height", "process grid height", 1);

  const string path_in = "blabla"; //Input<string>("--input", "path to the sequence");
  const string mode = "R"; //Input<string>("--mode", "[A]pprox or [R]andom");
  const string opt  = "N"; //Input<string>("--opt",  "[N]o, [S]ingle or [M]ultiple");

  if (argc == 1)
    {
      //ProcessInput();      
      //PrintInputReport();
    }

  FILE* degstream = NULL;
 
  int int_mode, int_opt;
  int* degrees = NULL;
  
  //Grid g(mpi::COMM_WORLD, height);
  //int commRank = MPI_Rank(MPI_COMM_WORLD);

  const int blk = nev + nex;
  //DistMatrix<C> A(N, N, g), B(N, N, g), V(N, blk, g), W(N, blk, g);
  //DistMatrix<R, VR, STAR> Lambda(blk, 1, g);  
  C *A, *B, *V, *W, *Lambda;
  A= new MKL_Complex16[N*N];
  B= new MKL_Complex16[N*N];
  V= new MKL_Complex16[N*blk];
  W= new MKL_Complex16[N*blk];
  Lambda= new MKL_Complex16[N];
  

  int filtered;
  int iterations;

  srand(0);
  //SetBlocksize(algbs);

  switch (mode[0])
    {
    case 'a':case 'A':
      int_mode = OMP_APPROX;
      break;
    case 'r':case 'R':
      int_mode = OMP_RANDOM;
      break;
    default:      
      std::cout << "Wrong mode argument." << std::endl;
    }

  switch (opt[0])
    {
    case 'n':case 'N':
      int_opt = OMP_NO_OPT;
      break;
    case 's':case 'S':
      int_opt = OMP_OPT;
      degrees = new int[nev];
      break; 
   /* case 'm':case 'M':
      int_opt = ELECHFSI_OPT_MULTIPLE;
      degrees = new int[nev];
      break; */
    default:     
      std::cout << "Wrong opt argument." << std::endl;
    }

  if (degrees != NULL)
    for (int i = 0; i < nev; ++i)
      degrees[i] = deg;

  //if (commRank == 0)
    std::cout << endl
              << setw(8) << "n "      << setw(7) << N     << setw(20) << ""
              << setw(8) << "algbs "  << setw(7) << algbs << endl
              << setw(8) << "nev "    << setw(7) << nev   << setw(20) << ""
            //  << setw(8) << "width "  << setw(7) << g.Width() << endl
              << setw(8) << "nex "    << setw(7) << nex   << setw(20) << ""
              << setw(8) << "height " << setw(7) << height<< endl
              << setw(8) << "deg "    << setw(7) << deg   << setw(20) << ""
              << setw(8) << "mode "   << setw(7) << mode  << endl
              << setw(8) << "tol "    << setw(7) << tol   << setw(20) << ""
              << setw(8) << "opt "    << setw(7) << opt   << endl
             // << setw(8) << "delta"   << setw(7) << get_delta()  << setw(20) << ""
             // << setw(8) << "degmax"  << setw(7) << get_degmax() << endl << endl
              << setw(8) << "path "   << path_in    << endl << endl;
  
  //   if (commRank == 0)
  //     std::cout << setw(3)  << "ell" 
  //               << setw(6)  << "iters"
  //               << setw(10) << "filtered"
  //               << setw(10) << "total"
  //               << setw(10) << "lanczos"
  //               << setw(10) << "filter"
  //               << setw(10) << "qr"
  //               << setw(10) << "reduced"
  //               << setw(10) << "conv"
  //               << endl;
  
  std::ostringstream problem(std::ostringstream::ate);



  // TESTING THE INTERFACE.

  //initialize the interface
  init(N); 

  problem << path_in << "gmat  1 " << setw(2) << bgn-1 << ".bin";
  read_matrix(A, N, (char*)problem.str().c_str());
  
  problem.clear(); problem.str("");  
  //MakeIdentity(B);
  for (int i=0; i<N*N; i++){
	  B[i].real=0;
	  B[i].imag=0;
  }
  for (int i=0; i<N*N; i+=N+1){
	  B[i].real=1;
	  B[i].imag=0;
  }

  set_AB(A, B);
  
  time_t starttime, endtime;

  // Use direct eigensolver for the first iteration.
  fl_el_diagonalize(nev, 1, nex, 0, 0.0, 0, 0);

  for (int i = bgn; i <= end ; ++i)
    {
      // Solve current problem.
      problem << path_in << "gmat  1 " << setw(2) << i << ".bin";
      read_matrix(A, N, (char*)problem.str().c_str());
      problem.clear(); problem.str("");

      set_AB(A, B);
      
	  time (&starttime);

      //double t = mpi::Time();
      fl_el_diagonalize(nev, 0, nex, deg, tol, int_mode, int_opt);
      //t = mpi::Time() - t;
      //if (commRank == 0)
      time (&endtime);
	  double dif = difftime (endtime,starttime);
	  std::cout << i << "\t"; //<< t << endl;
	  printf (" %.2lf ", dif );

    }
  
  //Finalize();
  return 0;
}
