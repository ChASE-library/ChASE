#include "El.hpp"
using namespace El;
using namespace std;

#include <string>
#include <iostream>
#include <sstream> // std::ostringstream
#include <fstream>
#include <iomanip> // std::setfill, std::setw, std::setprecision
#include <cstdio>
//#include <mkl.h>
typedef double Real;
typedef Complex<Real> C;

#include "../include/chase.hpp"
#include "../include/io.hpp"


int main(int argc, char* argv[])
{
  Initialize(argc, argv);
  mpi::Comm comm = mpi::COMM_WORLD;
  const int commRank = mpi::Rank(comm);

  try
  {
      const int  N   = Input<int>("--n",   "problem size");
      const int  nev = Input<int>("--nev", "number of wanted eigenpairs");
      const int  nex = Input<int>("--nex", "extra block size", 40);
      const int  deg = Input<int>("--deg", "degree", 25);
      const int  bgn = Input<int>("--bgn", "ell, begin");
      const int  end = Input<int>("--end", "ell, end");
      const Real tol = Input<Real>("--tol", "tolerance", 1e-10);

      const int algbs  = Input("--algbs",  "algorithmic block size", 128);
      const int height = Input("--height", "process grid height", 1);

      const string path_in = Input<string>("--input", "path to the sequence");
      const string mode = Input<string>("--mode", "[A]pprox or [R]andom","A");
      const string opt =
        Input<string>("--opt",  "[N]o, [S]ingle or [M]ultiple","N");
      const string path_eigp = Input<string>
        ("--eigp", "path to the (approximate) eigenpairs",  string(""));
      const string degfile = Input<string>
        ("--degfile", "filename for elechfsi_degrees_ell", string(""));

      ProcessInput();
      PrintInputReport();

      FILE* degstream = NULL;

      int int_mode=ELECHFSI_APPROX, int_opt=ELECHFSI_NO_OPT;
      int* degrees = NULL;

      const Grid g(comm, height);

      const int blk = nev + nex;
      DistMatrix<C> H(N, N, g), V(N, blk, g);
      DistMatrix<Real, VR, STAR> Lambda(blk, 1, g);

      Real* resid = new Real[nev];
      Real* time  = new Real[6];
      Real residual;
      int filtered;
      int iterations;

      if (degfile != "" && commRank == 0)
          degstream = fopen(degfile.c_str(), "wb");

      srand(0);
      SetBlocksize(algbs);

      switch (mode[0])
      {
        case 'a':case 'A':
          int_mode = ELECHFSI_APPROX;
          break;
        case 'r':case 'R':
          int_mode = ELECHFSI_RANDOM;
          break;
        default:
          LogicError("Invalid mode argument");
      }

      switch (opt[0])
      {
        case 'n':case 'N':
          int_opt = ELECHFSI_NO_OPT;
          break;
        case 's':case 'S':
          int_opt = ELECHFSI_OPT_SINGLE;
          degrees = new int[nev];
          break;
        case 'm':case 'M':
          int_opt = ELECHFSI_OPT_MULTIPLE;
          degrees = new int[nev];
          break;
        default:
          LogicError("Invalid opt argument");
      }

      if (degrees != NULL)
          for (int i = 0; i < nev; ++i)
              degrees[i] = deg;

      if (commRank == 0)
        std::cout
          << endl
          << setw(8) << "n "      << setw(7) << N     << setw(20) << ""
          << setw(8) << "algbs "  << setw(7) << algbs << endl
          << setw(8) << "nev "    << setw(7) << nev   << setw(20) << ""
          << setw(8) << "width "  << setw(7) << g.Width() << endl
          << setw(8) << "nex "    << setw(7) << nex   << setw(20) << ""
          << setw(8) << "height " << setw(7) << height<< endl
          << setw(8) << "deg "    << setw(7) << deg   << setw(20) << ""
          << setw(8) << "mode "   << setw(7) << mode  << endl
          << setw(8) << "tol "    << setw(7) << tol   << setw(20) << ""
          << setw(8) << "opt "    << setw(7) << opt   << endl
          << setw(8) << "delta"   << setw(7) << get_delta()  << setw(20) << ""
          << setw(8) << "degmax"  << setw(7) << get_degmax() << endl << endl
          << setw(8) << "path "   << path_in    << endl << endl;

      if (commRank == 0)
        std::cout
          << setw(3)  << "ell"
          << setw(6)  << "iters"
          << setw(10) << "filtered"
          << setw(10) << "total"
          << setw(10) << "lanczos"
          << setw(10) << "filter"
          << setw(10) << "qr"
          << setw(10) << "reduced"
          << setw(10) << "conv"
          //          << setw(10) << "resid"
          << endl;

      std::ostringstream problem(std::ostringstream::ate);


      for (int i = bgn; i <= end ; ++i)
      {
          if (path_eigp == "" && int_mode == ELECHFSI_APPROX)
          {
              // Solve previous problem.
              problem << path_in << "gmat  1 " << setw(2) << i-1 << ".bin";
              read_matrix(&H, (char*)problem.str().c_str());
              problem.clear(); problem.str("");

              HermitianEigSubset<Real> subset;
              subset.indexSubset = true;
              subset.lowerIndex  = 0;
              subset.upperIndex  = blk-1;
              HermitianEig(UPPER, H, Lambda, V, ASCENDING, subset);
          }
          else if (int_mode == ELECHFSI_APPROX)
          {
              // Read solutions to previous problem.
              problem << path_eigp << "gmat  1 " << setw(2) << i-1 << ".vct";
              read_matrix(&V, (char*)problem.str().c_str());
              problem.clear(); problem.str("");

              problem << path_eigp << "gmat  1 " << setw(2) << i-1 << ".vls";
              read_matrix(&Lambda, (char*)problem.str().c_str());
              problem.clear(); problem.str("");
          }
          else
          {
              MakeUniform(V);
              Zeros(Lambda, blk, 1);
          }

          // Solve current problem.
          problem << path_in << "gmat  1 " << setw(2) << i << ".bin";
          read_matrix(&H, (char*)problem.str().c_str());
          problem.clear(); problem.str("");

          chase(UPPER, H, V, Lambda, nev, nex, deg,
                degrees, tol, resid, int_mode, int_opt);

          // Full residual
          /*
          auto H_tmp = H( IR(1,blk), IR(1,blk) );
          DistMatrix<C> V_ref(N, blk, g);
          Gemm(NORMAL, NORMAL, C(1.0), H, V, C(0.0), V_ref);
          Gemm(ADJOINT, NORMAL, C(1.0), V, V_ref, C(0.0), H_tmp);
          UpdateRealPartOfDiagonal( H_tmp, -1, Lambda);
          residual = Norm( H_tmp, TWO_NORM );
          */

          // Store the output.
          // Sort(Lambda, V);
          //       problem << "gmat  1 " << setw(2) << i << ".vct";
          //       write_matrix(&V, (char*)problem.str().c_str());
          //       problem.clear(); problem.str("");
          //       problem << "gmat  1 " << setw(2) << i << ".vls";
          //       write_matrix(&Lambda, (char*)problem.str().c_str());
          //       problem.clear(); problem.str("");

          if (degfile != "" && commRank == 0)
          {
              int* tmp = new int[nev];
              int tmp2;
              get_degrees(tmp, &tmp2);
              fwrite(tmp, sizeof(int), nev, degstream);
          }

          // Print computation times and other.
          get_times(time);
          filtered   = get_filtered();
          iterations = get_iteration();

          if (commRank == 0)
            std::cout
              << setw(3)  << i
              << setw(6)  << iterations
              << setw(10) << filtered
              << setw(10) << setprecision(3) << time[0]
              << setw(10) << setprecision(2) << time[1]
              << setw(10) << setprecision(3) << time[2]
              << setw(10) << setprecision(2) << time[5]
              << setw(10) << setprecision(2) << time[3]
              << setw(10) << setprecision(2) << time[4]
              //              << setw(10) << setprecision(3) << residual
              << endl;
      }

      if (commRank == 0)
      {
          std::cout << endl << endl;
          PrintVersion();
          PrintConfig();
          PrintCCompilerInfo();
          PrintCxxCompilerInfo();
      }

      delete[] resid;
      delete[] time;
      if (degfile != "" && commRank == 0)
          fclose(degstream);
  }
  catch( exception& e ) { ReportException(e); }

  Finalize();
  return 0;
}
