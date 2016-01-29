#ifndef ELECHFSI_FLEUR
#define ELECHFSI_FLEUR

#include "El.hpp"
using namespace std;
using namespace El;

#include "chase.hpp"

typedef double R;
typedef Complex<R> C;

#ifdef CPP_INVERSION
typedef double F;
#else
typedef Complex<double> F;
#endif

class global_data
{
public:
  Grid *g;
  mpi::Comm mpi_comm;
  int matrix_dimension;
  DistMatrix<F> *H_mat,*S_mat;
  DistMatrix<F,STAR,VC> *eigenvectors;
  DistMatrix<R,VC,STAR> *eigenvalues;

  // Used for (the optimization of) the iterative eigensolver.
  // 1D arrays of length no_of_eigenvalues.
  int* degrees;
};

DistMatrix<F>* fleur_matrix(int n, F* buffer);

extern "C"
{
  void fl_el_initialize(int n, F* hbuf, F* sbuf, int mpi_used_comm);
  void fl_el_diagonalize(int no_of_eigenpairs, //int direct,
                         int nex, int deg, R tol, int mode, int opt);
  void fl_el_eigenvalues(int neig, R* eig);
  void fl_el_eigenvectors(int neig, R* eig, F* eigvec);
} // extern "C"

void set_AB(DistMatrix<F> &A, DistMatrix<F> &B);
void init_gd(Grid &g, int N);

#endif // ELECHFSI_FLEUR
