#include "El.hpp"
using namespace std;
using namespace El;

typedef double Real;
typedef Complex<Real> C;
typedef C F;

#include "../include/filter.hpp"
//#include "../include/io.hpp"


int main(int argc, char* argv[])
{
  Initialize(argc, argv);

  try
  {
      const int N   = Input<int>("--n",   "problem size");
      const int nev = Input<int>("--nev", "number of wanted eigenpairs");
      const int nex = Input<int>("--nex", "extra block size", 40);
      const int deg = Input<int>("--deg", "degree", 25);

      const int algbs  = Input("--algbs",  "algorithmic block size", 128);
      const int height = Input("--height", "process grid height", 1);

      //const string path_in = Input<string>("--input", "path to the sequence");

      ProcessInput();
      //PrintInputReport();

      //FILE* degstream = NULL;

      const Grid g(mpi::COMM_WORLD, height);

      const int blk = nev + nex;
      DistMatrix<C> A(N, N, g), V(N, blk, g), W(N, blk, g);

      srand(0);
      SetBlocksize(algbs);

      // std::ostringstream problem(std::ostringstream::ate);
      // problem << path_in;
      // read_matrix(&H, (char*)problem.str().c_str());
      // problem.clear();
      // problem.str("");

      OneTwoOne(A, N);
      Identity(V, N, blk);
      MakeUniform(W);

      auto applyA =
        [&]( F alpha, const ElementalMatrix<F>& X, F beta, ElementalMatrix<F>& Y )
        {
            Hemm( LEFT, UPPER, alpha, A, X, beta, Y );
        };

      const int numFilt = filter( applyA, V, W, 0, blk, deg, NULL, 0, 1., 1., 4. );
      if( mpi::WorldRank() == 0 )
          Output("Filtered ",numFilt," vectors");
  }
  catch(exception& e) { ReportException(e); }

  Finalize();
  return 0;
}
