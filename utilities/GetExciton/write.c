#include <complex>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

struct f_complex{
  float real;
  float imag;
};



void writefile( const char* filename, f_complex * arr, int m, int n){
  unsigned long mat_size = m*n;
  unsigned long count = 0;
  unsigned long o = m;
  unsigned long p = n;
  cout << "Writing Matrix of size " << m << " * " << n << endl;
  complex<double> * mat = new complex<double>[mat_size];
  for(unsigned long i = 0; i < o; i++){
      cout << "writing arr["<<i<< "]" << endl;
    for(unsigned long j = 0; j < p; j++){
      if(i == 20658)
	cout << "arr["<<i<<"]["<<j<<"] - " << i*m+j<< endl;
      mat[i*o+j] = complex<double>(arr[i*o+j].real, arr[i*o+j].imag);
      if(arr[i*o+j].real == 0.0 && arr[i*o+j].imag == 0)
	count++;
    }
  }
  cout << "Nonzero elements: " << count << " of " << mat_size << "(" << count/mat_size*100 << ")" << endl;
  cout << "Finished copying struct to datatype complex<double>" << endl;
  FILE *out = fopen(filename, "wb");
  if (out == NULL){
    fprintf( stderr, "Couldn't open file '%s'.\n", filename );
    exit( -1 );
  }
  unsigned long written = fwrite( mat, sizeof(double _Complex), m, out ); 
  if( written != m )
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
