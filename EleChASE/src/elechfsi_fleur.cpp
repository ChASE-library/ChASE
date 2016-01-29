/*
   Copyright (c) 2014, Daniel Wortmann
   All rights reserved.

   This file provides an interface from FLEUR to Elemental
*/

#include "../include/elechfsi_fleur.h"

static global_data *gd;

DistMatrix<F>* fleur_matrix(int n, F* buffer)
{
  // Create the distributed matrix
  DistMatrix<F,STAR,VC> *mat;
  // The Matrix should be a n x n matrix in 1-D cyclic distribution
  mat = new DistMatrix<F,STAR,VC>(n,n,*(gd->g));

  F* localbuffer = mat->Buffer(); // this is the local buffer of the matrix

  int local_index = 0;
  int fleur_index = 0;

  // Now copy all the data into local buffer to initialize the matrix
  // Loop over columns of local data
  for (int i = mpi::Rank(gd->mpi_comm); i < n; i += mpi::Size(gd->mpi_comm))
    {
      for (int j = 0; j <= i; j++)
        {
          // Use memcopy
          localbuffer[local_index++] = buffer[fleur_index++];
        }
      // further Off-diagonal elements are set to zero !Probably not needed
      for (int j = 0; j < n-i-1; j++)
        {
          localbuffer[local_index++] = 0.0;
        }
    }

  DistMatrix<F> *mat2 = new DistMatrix<F>(*mat);
  delete mat;
  return mat2;
}


extern "C"
{
  void fl_el_initialize(int n, F* hbuf, F* sbuf, int mpi_used_comm)
  // Set the two matrices
  {
    // Initialize the Library
    int argc = 0; char** argv;
    Initialize(argc, argv);

    //Store the matrix dimension & the mpi_communicator
    gd = new global_data;
    gd->mpi_comm = MPI_Comm_f2c(mpi_used_comm);
    gd->matrix_dimension = n;

    // First we need a mpi-grid
    gd->g= new Grid(gd->mpi_comm);

    // Store the Matrices
    gd->H_mat = fleur_matrix(n, hbuf);
    //    Display(*gd->H_mat);
    gd->S_mat = fleur_matrix(n, sbuf);

    gd->degrees = NULL;
    return;
  }


  void fl_el_diagonalize(int no_of_eigenpairs, //int direct,
                         int nex, int deg, R tol, int mode, int opt)
  // Diagonalize the Matrix (and return the number of local eigenvalues).
  // If direct = 1, Elemental's direct eigensolver is called, otherwise
  // EleChFSI is called.
  // *** JW: direct mode is currently disabled ***
  // Note that EleChFSI with mode = 0 (ELECHFSI_APPROX) requires the previous
  // eigenvectors and (some) eigenvalues to be available in the gd variable.
  {
    DistMatrix<R, VR, STAR> eigenval(no_of_eigenpairs+nex, 1, *(gd->g)), diag_view(*(gd->g));
    DistMatrix<F> evec(gd->matrix_dimension, no_of_eigenpairs+nex, *(gd->g)), mat_view(*(gd->g));

    // Initialize eigenvalues and eigenvectors, otherwise they are Matrices of size 0x0
    gd->eigenvalues = new DistMatrix<R,VC,STAR>(no_of_eigenpairs+nex, 1, *(gd->g));
    gd->eigenvectors = new DistMatrix<F,STAR,VC>(gd->matrix_dimension, no_of_eigenpairs+nex, *(gd->g));

    Cholesky(UPPER , *(gd->S_mat));
    TwoSidedTrsm(UPPER, NON_UNIT, *(gd->H_mat), *(gd->S_mat));


    //DistMatrix<F> evec2(gd->matrix_dimension, no_of_eigenpairs+nex, *(gd->g));
    double* resid = new double[no_of_eigenpairs];

    // Set approximate eigenvalues.
    // Here we need to use the new operator=
    // Casting over a View is no longer helpful
    eigenval = *(gd->eigenvalues);

    // Set approximate eigenvectors.
    // Here we need to use the new operator=
    evec = *(gd->eigenvectors);

    if(gd->degrees == NULL && opt != ELECHFSI_NO_OPT)
      {
	gd->degrees = new int[no_of_eigenpairs];
	for(int i = 0; i < no_of_eigenpairs; ++i)
	  gd->degrees[i] = deg;
      }

    HermitianEigSubset< Base<F> > subset;
    subset.lowerIndex = 0;
    subset.upperIndex = no_of_eigenpairs+nex;

    //HermitianEig(UPPER, *(gd->H_mat), eigenval, evec, ASCENDING, subset );
    chase(UPPER, *(gd->H_mat), evec, eigenval, no_of_eigenpairs,
    	  nex, deg, gd->degrees, tol, resid, mode, opt);

    //    Display( eigenval );

    delete[] resid; // Or maybe store it in gd.

    // Same as above, use operator=
    *(gd->eigenvalues)  = eigenval;
    *(gd->eigenvectors) = evec;
    return;
  }

  /*
  void fl_el_eigenvalues(int neig, R* eig)
  // Return the eigenvalues.
  {
    R* buf = gd->eigenvalues.Buffer();

    if (neig > gd->eigenvalues.LocalWidth()*gd->eigenvalues.LocalHeight())
      {
        cerr << "Error in dimensions in fleur_elemental\n";
      }

    for (int i = 0; i < neig; i++)
      {
        eig[i] = buf[i];
      }

    return;
  }
*/

  void fl_el_eigenvectors(int neig, R* eig, F* eigvec)
  // Return all the local eigenvectors & eigenvalues
  {
    R* eigbuf  = gd->eigenvalues->Buffer();
    F* eigbuff = gd->eigenvectors->Buffer();
    int local_index = 0;

    // Display(gd->eigenvalues);
    // Display(gd->eigenvectors);

    for (int i = 0; i < neig; i++)
      {
        //Copy eigenvalue
        eig[i] = eigbuf[i];

        //Copy eigenvector
        for (int j = 0; j < gd->matrix_dimension; j++)
          {
            eigvec[local_index] = eigbuff[local_index];
            local_index++;
          }
      }

    // Free the DistMatrix allocated with new
    delete gd->eigenvectors;
    delete gd->eigenvalues;
    return;
  }

} // extern "C"


/*
void set_AB(DistMatrix<F> &A, DistMatrix<F> &B)
{
  gd->H_mat = &A;
  gd->S_mat = &B;
  return;
}

void init_gd(Grid &g, int N)
{
  gd = new global_data;
  gd->mpi_comm = g.Comm();
  gd->matrix_dimension = N;
  gd->g= &g;
  gd->degrees = NULL;
  //  gd->eigenvalues  = DistMatrix<R,VC,STAR>(g);
  //  gd->eigenvectors = DistMatrix<F,STAR,VC>(g);
  return;
}
*/
