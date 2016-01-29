#ifndef ELECHFSI_FLEUR
#define ELECHFSI_FLEUR

//#include "elemental.hpp"

//using namespace elem;

#include "../../phiChASE/include/chfsi.h"
#include <mpi.h>

using namespace std;

typedef MKL_Complex16 C;
typedef double R;

#ifdef CPP_INVERSION
typedef double F;
#else
typedef MKL_Complex16 F;
#endif

class global_data
{
public:
  //Grid *g;
  MPI_Comm mpi_comm;
  int matrix_dimension;
  int no_of_eigenpairs;
  F *H_mat,*S_mat;
  F *eigenvectors;
  R *eigenvalues;

  // Used for (the optimization of) the iterative eigensolver.
  // 1D arrays of length no_of_eigenvalues.
  int* degrees;
};

F* fleur_matrix(int n, F* buffer);

extern "C"
{
  void fl_el_initialize(int n, F* hbuf, F* sbuf, int mpi_used_comm);
  void fl_el_diagonalize(int no_of_eigenpairs, int direct,
                         int nex, int deg, R tol, int mode, int opt);
  void fl_el_eigenvalues(int neig, R* eig);
  void fl_el_eigenvectors(int neig, R* eig, F* eigvec);
} // extern "C"

void set_AB(F *A, F *B);
void init(int N);


#endif // ELECHFSI_FLEUR
