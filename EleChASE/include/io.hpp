/*! \file io.hpp
 * 	\brief Header file for the IO functions
 * 	\details This file contains auxiliary IO functions, that read and write matrices from binary format, anddistribute them in Elemental style.
 *  */ 



#ifndef ELECHFSI_IO
#define ELECHFSI_IO

#include "El.hpp"
using namespace El;
using namespace std;


/** \fn read_matrix(ElementalMatrix<F>* A, char* filename)
 * \brief Reads matrix in a binary format, and puts the data in Elemental DistMatrix.
 * \param A			Elemental DistMatrix of template type F. Used to store the input Hermitian matrix
 * \param filename		Pointer to an array of characters. Contains the name of the binary file that contains the matrix.
 * \return void
 */
template<typename F>
void read_matrix(ElementalMatrix<F>* A, char* filename)
{
  const int c_shift  = A->ColShift();  // first row we own
  const int r_shift  = A->RowShift();  // first col we own
  const int c_stride = A->ColStride();
  const int r_stride = A->RowStride();
  const int l_height = A->LocalHeight();
  const int l_width  = A->LocalWidth();
  F tmp;

  FILE* stream = fopen(filename, "rb");
  if (stream == NULL)
    {
      cerr << "Couldn't open file " << string(filename) << endl;
      exit(-1);
    }

  for (int lj = 0; lj < l_width; ++lj)    
    for (int li = 0; li < l_height; ++li)
      {
	// Our process owns the rows c_shift:c_stride:n,
	// and the columns           r_shift:r_stride:n.
	int i = c_shift + li*c_stride;
	int j = r_shift + lj*r_stride;
        
	fseek(stream, (j*A->Height()+i)*sizeof(F), SEEK_SET);
	fread(reinterpret_cast<void*>(&tmp), sizeof(F), 1, stream);
	
	A->SetLocal(li, lj, tmp);
      }
    
  fclose(stream);
  mpi::Barrier(A->Grid().Comm());
  return;
}
 
/** \fn write_matrix(ElementalMatrix<F>* A, char* filename)
 * \brief Outputs an Elemental DistMatrix in a binary format.
 * \param A			Elemental DistMatrix of template type F. 
 * \param filename		Pointer to an array of characters. Contains the name of the binary file where the matrix is to be written.
 * \return void
 */
template<typename F,Dist XColDist,Dist XRowDist>
void write_matrix
(DistMatrix<F, XColDist, XRowDist>* A, 
 char* filename)
{
  FILE* stream = NULL;
  const int height = A->Height();
  const int width  = A->Width();
  DistMatrix<F, XColDist, XRowDist> col_view(A->Grid());
  DistMatrix<F, STAR, STAR> col_star(height, 1, A->Grid());

  if (A->Grid().Rank() == 0)
    { 
      stream = fopen(filename, "wb");
      if (stream == NULL)
	{
	  cerr << "Couldn't open file " << string(filename) << endl;
	  exit(-1);
	}
    }
 
  // Writes the matrix column-per-column.
  for (int i = 0; i < width; ++i)
    {
      View(col_view, *A, 0, i, height, 1);
      col_star = col_view;
      if (A->Grid().Rank() == 0 && 
	  fwrite(col_star.Buffer(), sizeof(F), height, stream) != height)
	{
	  fprintf( stderr, "Didn't write all the data.\n" );
	  exit( -2 );
	}      
    }

  if (A->Grid().Rank() == 0) fclose(stream);
  mpi::Barrier(A->Grid().Comm());
  return;
}

#endif //ELECHFSI_IO
