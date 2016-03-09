#include <iostream>
#include <vector>
#include <complex.h>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <mkl.h>


using namespace std;

struct matrices{
  string name;
  string lambda;
  string evec;
  int size;
};

template<typename T>
void write2file( string &filename, T * mat, int m, int n){
  FILE *out = fopen(filename.c_str(), "wb");
  if (out == NULL){
    fprintf( stderr, "Couldn't open file '%s'.\n", filename.c_str() );
    exit( -1 );
  }
  if( fwrite( mat, sizeof(T), m*n, out ) != m*n )
    {
      fprintf( stderr, "Didn't write all the data.\n" );
      exit( -2 );
    }
  fclose( out );
  return;
}

template<typename T>
void readMatrix(T *H, string path_in, string filename, int size){
  ostringstream problem(ostringstream::ate);
  problem << path_in << filename;
  ifstream input(problem.str().c_str(), ios::binary);
  
  if(input.is_open()){
    input.read((char *) H, sizeof(T) * size);
  } else {
    throw string("error reading file: ") + problem.str();
  }
}

vector<matrices> readList(string &in_dir, string &out_dir, int &max_size){
  ifstream file(in_dir + string("mat_sizes.txt"));
  if(!file.is_open())
    throw string("error reading matrices list");
  string line;
  vector<matrices> vec;
  matrices elem;
  max_size = 0;
  while(getline(file,line)) {
    stringstream linestream(line);
    string name;
    int size;
    getline(linestream, name, '\t');
    linestream >> size;
    elem.name = name;
    name.replace(name.end()-4, name.end(), ".vls");
    elem.lambda = out_dir + name;
    name.replace(name.end()-4, name.end(), ".vct");
    elem.evec = out_dir + name;
    elem.size = size;
    max_size = ( max_size<size ) ? size : max_size;
    vec.push_back(elem);
  }
  return vec;
}

int main(int argc, char* argv[]){
  if(argc < 2) {
    cout << "Please give a directory." << endl;
    exit(-1);
  }
  string in_dir = argv[1];
  string out_dir;
  if(argc < 3) {
    cout << "No output directory given. Using current working directory." << endl;
    out_dir = "./";
  } else {
    cout << "Writing files to: " << argv[2] << endl;
    out_dir = argv[2];
  }
  int max_size;
  vector<matrices> vec;
  try{
    vec = readList(in_dir, out_dir, max_size);
  } catch(exception &e){
    cout << e.what() << endl << endl;
  }
  if(vec.size() == 0){
    cout << "No files found." << endl;
    return 0;
  }

  double abstol = LAPACKE_dlamch('S');
  MKL_Complex16 * mat;
  lapack_int il = 1;
  lapack_int iu = max_size / 10;
  lapack_int m;
      
  for(unsigned int i = 0; i < vec.size(); i++){
    mat = new MKL_Complex16[vec[i].size * vec[i].size];
    cout << "Filename: " << vec[i].name << endl;
    readMatrix(mat, in_dir, vec[i].name, vec[i].size*vec[i].size);
    lapack_int * isuppz = new lapack_int[2*vec[i].size];
    double * Lambda = new double[vec[i].size];
    MKL_Complex16 * V      = new MKL_Complex16[vec[i].size * vec[i].size];
    LAPACKE_zheevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U', vec[i].size, mat, vec[i].size, 0.0, 1.0, il, iu, abstol, &m, Lambda, V, vec[i].size, isuppz);
    write2file(vec[i].lambda, Lambda, 1, m);
    write2file(vec[i].evec, V, vec[i].size, m);
    delete[] isuppz;
    delete[] Lambda;
    delete[] V;
    delete[] mat;
  }

  return 0;
}
