//Interface from FLEUR to (phi and OpenMP versions of) ChASE

//WORK IN PROGRESS, NOT USABLE!

#include "phichase_fleur.h"

static global_data *gd;

F* fleur_matrix(int n, F* buffer)
{
  // Create the distributed matrix
  //DistMatrix<F,STAR,VC> *mat;
  // The Matrix should be n x n matrix in 1-D cyclic distribution
  //mat = new DistMatrix<F,STAR,VC>(n,n,*(gd->g));

  F* localbuffer = new F[n]; //mat->Buffer(); // this is the local buffer of the matrix
  F* rootbuffer = new F[n*n];

  const int buffersize1 = n; // mat->LocalWidth();
  const int buffersize2 = n; //mat->LocalHeight();
  int local_index = 0;
  int fleur_index = 0;

  // Now copy all the data into local buffer to initialize the matrix
  // Loop over columns of local data
  
  
																		//check if the send and receive could work with array chunks
																		//CHECK IF IT SUPPORTS THE COMPLEX TYPE (can be problematic)
  int i,cs;
  MPI_Comm_rank(gd->mpi_comm, &i);
  MPI_Comm_size(gd->mpi_comm, &cs);
  
  for (; i < n; i += cs)
    {
      for (int j = 0; j <= i; j++)
        {
			//taking conjugate (because of the implicit transposing here)
          localbuffer[local_index].real=buffer[fleur_index].real;
          localbuffer[local_index++].imag=-buffer[fleur_index++].imag;
        }
      // further Off-diagonal elements are set to zero !Probably not needed
      for (int j = 0; j < n-i-1; j++)
        {
		  localbuffer[local_index].real=0;
          localbuffer[local_index++].imag=0;
        }
        
       // MPI_Send((localbuffer+local_index-n), n, MPI::DOUBLE_COMPLEX, 0, 0, gd->mpi_comm);
        
        
        //int number;
		//if (world_rank == 0) {
			//MPI_Recv((localbuffer+local_index-n), n, MPI::DOUBLE_COMPLEX, i, 0, gd->mpi_comm, MPI_STATUS_IGNORE);

		//}
    }
    
    MPI_Gather(localbuffer, n, MPI::DOUBLE_COMPLEX,rootbuffer,n*n,MPI::DOUBLE_COMPLEX,0,gd->mpi_comm);
    
    //filling the upper triangular part with the complex conjugates of the lower half
    for (int i=0; i<n; i++){
		int row=i*n;
		for (int j=i+1; j<n; j++){
			localbuffer[row+j].real=localbuffer[j*n+i].real;
			localbuffer[row+j].imag=-localbuffer[j*n+i].imag;
		}
	}

/*    
    MPI_Gather(
    void* send_data,
    int send_count,
    MPI_Datatype send_datatype,
    void* recv_data,
    int recv_count,
    MPI_Datatype recv_datatype,
    int root,
    MPI_Comm communicator)
*/
    
    //mpi gather should be here

  //DistMatrix<F> *mat2 = new DistMatrix<F>(*mat);
  //delete mat;
  return rootbuffer;//mat2;
}

void eigenSolve(int bgn, int end, int n, MKL_Complex16 * A,  MKL_Complex16 * Z, double* values) {

	//returns real lambdas always (double precision)

	char jobz = 'V'; //both eigenvalues and eigenvectors are computed
	char range = 'I';  //if problems arrise, use 'A' (all)	(otherwise 'I') 
	char uplo='L'; //May be 'U' as well. Paper says that L is faster
		
	//n is the size of the matrix (used in the algorithm)
		
	//A is used as an input for a matrix
		
	double vl,vu;	//vl, vu are discarded in our case (range =I)

	//beginning and end of the needed values
	int il=bgn+1;
	int iu=end+1;
	double abstol=10e-15; //constant. No need to be taken as input. Or use default
		
	double orfac=10e-3; // Default value used. If problems appear, check (info variable) and change
		
	//z, iz, jz, desc_z   :returned eigenvectors and related descriptors for them
	int iz=1, jz=1;		//ones
		
	//return values: m-num of eigenvalues, values (w)-the  eigenvalues, nz- number of eigenvectors
	int m,nz;	
		
	int lwork=-1;	//this set of parameters is needed for the empty scalapack call, in order to calculate the workspace
	
	MKL_Complex16 * work= new MKL_Complex16[n];
	double* rwork= new double[n];
	int* iwork=new int[n];
		
	//ifail, iclustr, info -error return outputs
	int info;

	int* ifail=new int[n];
	int* iclustr=new int[n];
	
	//empty call to the function, to calculate the work size
	zheevx_(&jobz, &range, &uplo, &n, A, &n, 
			&vl, &vu, &il, &iu, &abstol, &m, values, Z, &n,
			work, &lwork, rwork, iwork, ifail, &info); 
			
	//ZHEEVX( JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK, LWORK, RWORK, IWORK, IFAIL, INFO );
		
	cout<<" info1="<<info<<endl;
	

	lwork  =work[0].real;  

	work = (MKL_Complex16 *) malloc(lwork*sizeof(MKL_Complex16));
	//assert(work != NULL);

	rwork = (double *) malloc(7*n*sizeof(double));
	//assert(rwork != NULL);

	iwork = (int *) malloc(5*n*sizeof(int));
	//assert(iwork != NULL);
	
	double t1=0,t3=0; //time measurement
	
	t1 = MPI_Wtime();
	//call to the function	
	zheevx_(&jobz, &range, &uplo, &n, A, &n, 
			&vl, &vu, &il, &iu, &abstol, &m, values, Z, &n,
			work, &lwork, rwork, iwork, ifail, &info); 
			
	//ZHEEVX( JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK, LWORK, RWORK, WORK, IFAIL, INFO );
		
	t3 = MPI_Wtime();
		
	cout<<" info2="<<info<<endl;
	
	
	//if (myRank == 0) {
	//	cout<<setprecision(5)<<"Time="<<t3-t1<<endl;
	//}
	
	/*
	if (myRank==0){
		cout<<"values="<<m<<" vectors="<<nz<<endl;
		
		cout<<"failed to converge: "<<endl;
	for(int j=0;j<m;j++){
		  //values[j]=1;
		  cout<<ifail[j]<<" ";
	}
	
	cout<<endl<<"not orthogonalized: "<<endl;
	for(int j=0;j<m;j++){
		  //values[j]=1;
		  cout<<iclustr[j]<<" ";
	}
	cout<<endl;
	}
	*/
	return;	
}


extern "C"
{
  void fl_el_initialize(int n, F* hbuf, F* sbuf, int mpi_used_comm)				//seems ok so far
  // Set the two matrices
  {
    // Initialize the Library - no need of this in the phi version
    //int argc = 0; char** argv;
    //Initialize(argc, argv);

    //Store the matrix dimension & the mpi_communicator
    gd = new global_data;
    gd->mpi_comm = MPI_Comm_f2c(mpi_used_comm);
    gd->matrix_dimension = n;

    // First we need a mpi-grid
    //gd->g= new Grid(gd->mpi_comm);

    // Store the Matrices
    gd->H_mat = fleur_matrix(n, hbuf);
    gd->S_mat = fleur_matrix(n, sbuf);

    gd->degrees = NULL;
    return;
  }


  void fl_el_diagonalize(int no_of_eigenpairs, int direct,
                         int nex, int deg, R tol, int mode, int opt)
  // Diagonalize the Matrix (and return the number of local eigenvalues).
  // If direct = 1, Elemental's direct eigensolver is called, otherwise
  // EleChFSI is called.
  // Note that EleChFSI with mode = 0 (ELECHFSI_APPROX) requires the previous
  // eigenvectors and (some) eigenvalues to be available in the gd variable.
  {    
	  char uplo='L';
	  int info=0;
	  int itype=1;
	  int n=gd->matrix_dimension;
	  gd->no_of_eigenpairs=no_of_eigenpairs;
	  
	  R* eigenval= new R[n];
	  F* evec= new F[n*n];
	  F* evec2= new F[n*n];
	  
    //DistMatrix<R, VR, STAR> eigenval(no_of_eigenpairs+nex, 1, *(gd->g)), diag_view(*(gd->g));
    //DistMatrix<F> evec(gd->matrix_dimension, no_of_eigenpairs+nex, *(gd->g)), mat_view(*(gd->g));
      
																		// should find a solution to this	
    //Cholesky(UPPER , *(gd->S_mat));									//ZPOTRF( UPLO, N, A, LDA, INFO )
    //TwoSidedTrsm(UPPER, NON_UNIT, *(gd->H_mat), *(gd->S_mat));		//ZHEGST( ITYPE, UPLO, N, A, LDA, B, LDB, INFO )
    zpotrf_( &uplo, &n, gd->S_mat, &n, &info);
    zhegst_( &itype, &uplo, &n, gd->H_mat, &n, gd->S_mat, &n, &info);

    if(direct) {
        //HermitianEig(UPPER, *(gd->H_mat), eigenval, evec, 0, no_of_eigenpairs+nex, UNSORTED);
        eigenSolve(1, no_of_eigenpairs+1, n, gd->H_mat,  evec, eigenval);
         //ZHEEVX( JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK, LWORK, RWORK, WORK, IFAIL, INFO );
        // Look into the sort and HermitianEigCtrl.
    }
    else {
        F *evec2= new F[gd->matrix_dimension* no_of_eigenpairs+nex];
        double* resid = new double[no_of_eigenpairs];

		// Set approximate eigenvalues.
		//View(diag_view, eigenval, 0, 0, no_of_eigenpairs+nex, 1);
		//diag_view = gd->eigenvalues;

		// Set approximate eigenvectors.
		//View(mat_view, evec, 0, 0, gd->matrix_dimension, no_of_eigenpairs+nex);
        //mat_view = gd->eigenvectors;

        if(gd->degrees == NULL && opt != OMP_NO_OPT) {
            gd->degrees = new int[no_of_eigenpairs];
            for(int i = 0; i < no_of_eigenpairs; ++i)
				gd->degrees[i] = deg;
        }

       // chfsi(UPPER, *(gd->H_mat), evec, evec2, eigenval, no_of_eigenpairs, 
	    //  nex, deg, gd->degrees, tol, resid, mode, opt);
		//last variable (set to 0 here) is const int int_arch.
		//I think it is not actually used, so I entered a "random" value
		
	    chfsi(gd->H_mat, n, evec, evec2, eigenval, 
	    no_of_eigenpairs, nex, deg, tol, mode, opt, 0);

        delete[] resid; // Or maybe store it in gd.
    }
    //View(diag_view, eigenval, 0, 0, no_of_eigenpairs+nex, 1);
    //View(mat_view, evec, 0, 0, gd->matrix_dimension, no_of_eigenpairs+nex);
    gd->eigenvalues  = eigenval; //diag_view;
    gd->eigenvectors = evec; //mat_view;
    return;
  }


	void fl_el_eigenvalues(int neig, R* eig){			//seems ok so far
		// Return the eigenvalues.
  
		//R* buf = gd->eigenvalues.Buffer();

		if (neig > gd->no_of_eigenpairs){
			cerr << "Error in dimensions in fleur_elemental\n";
		}

		for (int i = 0; i < neig; i++){
			eig[i] = gd->eigenvalues[i];
		}

		return;
	}

	void fl_el_eigenvectors(int neig, R* eig, F* eigvec){
	// Return all the local eigenvectors & eigenvalues

		//R* eigbuf  = gd->eigenvalues.Buffer();
		//F* eigbuff = gd->eigenvectors.Buffer();
		int local_index = 0;

		/**/ //Display(gd->eigenvalues);
		/**/ //Display(gd->eigenvectors);

		for (int i = 0; i < neig; i++){
			//Copy eigenvalue
			//int pe = mpi::Rank(gd->mpi_comm);
			//int in = i*mpi::Size(gd->mpi_comm)+pe;

			/**/ //cout<< "PE:" << pe << ":" << i << "->" << in << endl;

			eig[i] = gd->eigenvalues[i];

			//Copy eigenvector
			for (int j = 0; j < gd->matrix_dimension; j++){
				eigvec[local_index] = gd->eigenvectors[local_index]; // supposing vectors are stored in row-major order in C
				local_index++;									
			}
		}

		return;
	}

} // extern "C"


  
void set_AB(F *A, F *B)			//seems ok so far
{
	gd->H_mat = A;
	gd->S_mat = B;
	return;
}

void init(int N)	//semms ok so far			
{
	gd = new global_data;
	//gd->mpi_comm = g.Comm();														//CHANGE THIS!!! ADD A COMM PARAMETER IN THE CALL!!!
	
	gd->matrix_dimension = N;
	//gd->mpi_comm= &c;
	gd->degrees = NULL;
	gd->eigenvalues  = new R[N];
	gd->eigenvectors = new F[N*N];
	return;
}

  
