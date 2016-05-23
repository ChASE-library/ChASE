#include <complex>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mkl.h>

using namespace std;

struct f_complex{
  float real;
  float imag;
};



void writefile( const char* filename, f_complex * arr, unsigned long m, unsigned long n){
  unsigned long mat_size = m*n;
  unsigned long count = 0;
  cout << "Writing Matrix of size " << m << " * " << n << endl;
  MKL_Complex16 * mat = new MKL_Complex16[mat_size];
  for(unsigned long i = 0; i < mat_size; i++){
    mat[i].real = arr[i].real;
    mat[i].imag = arr[i].imag;
    if(arr[i].real == 0.0 && arr[i].imag == 0.0)
      count++;
  }
  cout << "Zero elements: " << count << " of " << mat_size << "(" << (double)count/(double)mat_size*100 << ")" << endl;
  cout << "Finished copying struct to datatype complex<double>" << endl;
  FILE *out = fopen(filename, "wb");
  if (out == NULL){
    fprintf( stderr, "Couldn't open file '%s'.\n", filename );
    exit( -1 );
  }
  unsigned long written = fwrite( mat, sizeof(MKL_Complex16), m*n, out ); 
  if( written != m*n )
    {
      fprintf( stderr, "Didn't write all the data.\n" );
      exit( -2 );
    }
  delete[] mat;
  fclose( out );
  return;
}


extern"C" void write2file_(f_complex * array, int * m, int * n){
  writefile("mat.bin", array, *m, *n);
}
