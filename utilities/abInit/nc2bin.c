/*
module load intel-para/2015.07
module load netCDF-C++4/4.2.1

icpc nc2bin.c -o nc2bin -lnetcdf_c++4

nc2bin path_to_nc_file.nc
*/

#include <iostream>
#include <stdlib.h>
#include <string>
#include <netcdf>
#include <complex.h>
#include <fstream>

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

static const int NC_ERR = 2;

void write2file( const char* filename, double complex * mat, int m, int n){
  FILE *out = fopen(filename, "wb");
  if (out == NULL){
    fprintf( stderr, "Couldn't open file '%s'.\n", filename );
    exit( -1 );
  }
  if( fwrite( mat, sizeof(double _Complex), m*n, out ) != m*n )
    {
      fprintf( stderr, "Didn't write all the data.\n" );
      exit( -2 );
    }
  fclose( out );
  return;
}

int main(int argc, char * argv[]){
  try {
    if(argc < 2) {
      cout << "Path to File needed." << endl;
      exit(-1);
    }
    else {
      cout << "Setting input file as: " << argv[1] << endl;
    }
    string out_path;
    if(argc < 3) {
      cout << "No output directory given, using current working directory." << endl;
      out_path = "./";
    } 
    else {
      out_path = argv[2];
      cout << "Setting output path to: " << out_path << endl;
    }
    const string fileIn(argv[1]);
    NcFile ncFile(fileIn, NcFile::read);
    
    // Reading Dimensions
    int ncomplex   = ncFile.getDim("complex").getSize();
    int max_ncoeff = ncFile.getDim("max_number_of_coefficients").getSize();
    int nkps       = ncFile.getDim("number_of_kpoints").getSize();
    int nsteps     = ncFile.getDim("nstep").getSize();
    int nspin      = ncFile.getDim("number_of_spins").getSize();
    cout << ">>>>>>>>>>>>>> Dimensions <<<<<<<<<<<<<<" << endl;
    cout << "Complex dimensions: "         << ncomplex << endl;
    cout << "Max number of coefficients: " << max_ncoeff << endl;
    cout << "Number of k-points: "         << nkps << endl;
    cout << "Number of spins: "            << nspin << endl;
    cout << "Number of steps: "            << nsteps << endl;
    cout << endl;

    // Reading Variables
    int      last_step;
    int *    ncoeff        = new int[nkps];
    double * ghg           = new double[nsteps * nspin * nkps * max_ncoeff * max_ncoeff * ncomplex];
    NcVar    last_step_var = ncFile.getVar("last_step");
    NcVar    ncoeff_var    = ncFile.getVar("number_of_coefficients");
    NcVar    ghg_var       = ncFile.getVar("ghg");
    last_step_var.getVar(&last_step);
    ncoeff_var.getVar(ncoeff);
    ghg_var.getVar(ghg);
    cout << ">>>>>>>>>>>>>> Variables <<<<<<<<<<<<<<" << endl;
    cout << "Last step: " << last_step << endl;
    cout << "Number of coefficients: " << endl;
    for(int i = 0; i < nkps; i++) cout << i << ". k-point: " << ncoeff[i] << endl;
    cout << endl;

    // Loop over ghg and write matrices to file
    char str[200];
    double complex * mat;
    ofstream myfile;
    string out_file = out_path + "mat_sizes.txt";
    myfile.open(out_file.c_str());
    for(int ikps = 0; ikps < nkps; ikps++){
      mat = new double complex[ncoeff[ikps] * ncoeff[ikps]];
      for(int istep = 0; istep < last_step; istep++){
	for(int ispin = 0; ispin < nspin; ispin++){
	  int z = ncomplex * max_ncoeff * max_ncoeff * ( ikps + nkps * ( ispin + nspin * istep) );
	  for(int i = 0; i < ncoeff[ikps]; i++){
	    for(int j = 0; j < ncoeff[ikps]; j++){
	      mat[i*ncoeff[ikps] + j] = ghg[z + i*max_ncoeff*ncomplex + j*ncomplex + 0] + I*ghg[z + i*max_ncoeff*ncomplex + j*ncomplex + 1];
	    }
	  }
	  str[0] = '\0';
	  sprintf(str, "mat_%c_%02d_%02d.bin", (ispin == 1) ? 'u' : 'd', ikps, istep);
	  out_file = out_path + str;
	  cout << "Writing: " << str << endl;
	  myfile << str << "\t" << ncoeff[ikps] << endl;
	  write2file( out_file.c_str(), mat, ncoeff[ikps], ncoeff[ikps]);
	}
      }
      delete[] mat;
    }
    myfile.close();
    delete[] ghg;
    delete[] ncoeff;
    return 0;

  } catch(NcException &e) {
    e.what();
    cout << ">>>>>>>>>>>>>>FAILURE<<<<<<<<<<<<<<" << endl;
    return NC_ERR;
  }
}
