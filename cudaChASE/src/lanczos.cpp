#include "../include/lanczos.h"
#include <string.h>

void lanczos(MKL_Complex16 *B, MKL_Complex16 *v, int n, int blk, int m, double tol, int ctrl, double *ritzv, double *bound)
{  

  cout << "LANCZOS..." << endl;

  MKL_Complex16 *v0 = NULL; // Auxiliary vector.

  double *TeValues  = NULL; // Eigenvalues  of the tridiagonal matrix T.
  MKL_Complex16 *TeVectors = NULL; // Eigenvectors of the tridiagonal matrix T.
  MKL_Complex16 *t = NULL; // Used for reorthogonalization.

  double *d = NULL; // Contains the     diagonal elements of the tridiagonal matrix T.
  double *e = NULL; // Contains the off-diagonal elements of the tridiagonal matrix T.
  double *tmp_d = NULL; // Temporary storage for the array d.   
  double *tmp_e = NULL; // Temporary storage for the array e.  
  double *swap  = NULL; // Used for saving the tridiagonal matrix T.

  int tryrac   = 1; // Input parameter for dstemr_.
  MKL_Complex16 *zwork = NULL;

  int   *isuppz = NULL; // Input parameter for dstemr_. Not used (but needed).
  double vl, vu; // Input parameter for dstemr_. Not used (but needed).
  int    il, iu; // Input parameter for dstemr_. Not used (but needed).

  int lzwork;

  int notneeded_m; // Output parameter of dstemr_.

  double Tnorm, tmp; // Used for the computation of the bound.

  MKL_Complex16 alpha, alpha2;
  MKL_Complex16 beta;
  double real_alpha = 0.0;
  double real_beta  = 0.0;

  MKL_Complex16 *G     = NULL;

  int iONE = 1;
  int INFO = 0;

  int M = ( ctrl == 1 ) ? 4*blk : m;

  MKL_Complex16 *V = new MKL_Complex16[n*M]; // Transformation matrix consisted of Lanczos vectors.

  //  if( ctrl == 0)
    TeVectors = new MKL_Complex16[m*m];

  if( ctrl == 1 )
    {      
      lzwork = n > 2*M ? n : 2*M;
      G = new MKL_Complex16[n*n];
      zwork = new MKL_Complex16[lzwork];
    }
  //----------------------COMPUTATION-OF-THE-OPTIMAL-WORKSPACES-------------------------
  double *dwork = new double[1]; // Input parameter for dstemr_.
  int *iwork = new int[1]; // Input parameter for dstemr_.
  
  int ldwork = -1, liwork = -1;
  dstemr_( "V", "A", &M, d, e, &vl, &vu, &il, &iu, 
           &notneeded_m, TeValues, (double*)TeVectors, &M, &M, 
           isuppz, &iONE, dwork, &ldwork, iwork, &liwork, &INFO );

  ldwork = dwork[0];
  liwork = iwork[0];
  
  delete[] dwork; delete[] iwork;
      
  ldwork = ldwork > 24*M ? ldwork : 24*M;
  dwork = new double[ldwork];
  TeValues = new double[M];
  d = new double[2*m];
  e = new double[2*m];
  
  liwork = (liwork > 10*M) ? liwork : 10*M;
  iwork = new int[liwork];
  isuppz = new int[2*M];
  //--------------------------------------------------------------------------------------
  
  //------------------------------ alpha = v / ||v|| -------------------------------------
  real_alpha = DZNRM2( &n, (const MKL_Complex16*)v, &iONE );
 
  alpha = MKL_Complex16 (1/real_alpha, 0.0);
  ZSCAL( &n, &alpha, (MKL_Complex16*)v, &iONE );
  //--------------------------------------------------------------------------------------


  MKL_Complex16 *v1 = v; // Auxiliary vector
  // v0 = NULL;
  
  for(int k = 0; k < M ; )
    {
      //----------------------------- V(:,k) = v1 -----------------------------------------
      memcpy( V+k*n, v1, n * sizeof(MKL_Complex16) );
      //-----------------------------------------------------------------------------------

      //----------------------------- v = B*v1 - bet*v0 -----------------------------------
      v1 = V+k*n;

      alpha = MKL_Complex16 (1.0,0.0);
      beta = MKL_Complex16 (0.0, 0.0);
          
      // v := alpha*B*v1 + beta*v
      ZGEMV( "N", &n, &n, &alpha, (const MKL_Complex16*)B, &n, 
             (const MKL_Complex16*)v1, &iONE, &beta, v, &iONE );

      //------------------------------------------------------------------------------------
 
      if( v0 != NULL )
        {          
	  beta = MKL_Complex16 (-real_beta,0.0);
          ZAXPY( &n, &beta, (const MKL_Complex16*)v0, &iONE, v, &iONE );
        }

      // alp(ha) = v1'*v;
      ZDOTC( &alpha, &n, (const MKL_Complex16*)v1, &iONE, (const MKL_Complex16*)v, &iONE );
      //------------------------------------------------------------------------------------

      //------------------------------- v = alpha*v1 + v -----------------------------------
      alpha *= MKL_Complex16 (-1.0,0.0);
      
   
      ZAXPY( &n, &alpha, (const MKL_Complex16*)v1, &iONE, v, &iONE );
      //------------------------------------------------------------------------------------

      //----------------------------------RESTORE-alpha-------------------------------------
      alpha *= MKL_Complex16 (-1.0,0.0);
      //------------------------------------------------------------------------------------      

      //--------------------------------- beta = norm(v) -----------------------------------
      real_beta = DZNRM2( &n, (const MKL_Complex16 *)v, &iONE );
      //------------------------------------------------------------------------------------

      v0 = v1;

      //----------------------------- v1 = v/beta; (v = v/beta) ----------------------------
      beta = MKL_Complex16 (1.0/real_beta,0.0);
      
      ZSCAL( &n, (const MKL_Complex16*)(&beta), v, &iONE );
      v1 = v;
      //------------------------------------------------------------------------------------

      if( k < m )
        { // Needed for the upper bound.
	  d[k] = real(alpha);
          e[k] = real_beta;
        }

      k += 1;
      if( k == m ) // Compute the upper bound.
        {                   
          if( ctrl == 1 ) // RANDOM: Compute eigenvalues only.
            {         
              DSTEMR( "N", "A", &k, d, e, &vl, &vu, &il, &iu, 
                      &notneeded_m, TeValues, (double*)TeVectors, &M, &k, 
                      isuppz, &tryrac, dwork, &ldwork, iwork, &liwork, &INFO );               
            }
          else // APPROX: Compute eigenvalues and eigenvectors.
            {        
              DSTEMR( "V", "A", &k, d, e, &vl, &vu, &il, &iu, 
                      &notneeded_m, TeValues, (double*)TeVectors, &M, &k, 
                      isuppz, &tryrac, dwork, &ldwork, iwork, &liwork, &INFO );       
            }
	  // Find maximal absolute value of all obtained eigenvalue approximations.
	  // They are sorted so the values furthest from 0 are first and last.
          Tnorm = ( fabs( TeValues[0] ) > fabs( TeValues[k-1] ) ) 
            ? fabs( TeValues[0] ) : fabs( TeValues[k-1] );        

          if( ctrl == 1 ) // RANDOM
            *bound = Tnorm + fabs(real_beta);         
          else // APPROX
            { 
              *bound = Tnorm + fabs(real_beta)*fabs(((double*)TeVectors)[(k-1)*M+k-1]); 
	      cout << "LANCZOS done." << endl;

              return;
            }

        } // if( k == m )

    } // for(int k = 0; k < M ; )
  
  memcpy( G, B, n*n * sizeof(MKL_Complex16) );
  
  // [V, ~ ] = qr(V);
  ZGEQRF( &n, &M, V, &n, (MKL_Complex16*)dwork, zwork, &lzwork, &INFO ); 
  // G = V'B
  ZUNMQR( "L", "C", &n, &n, &M, V, &n, (MKL_Complex16*)dwork, 
          G, &n, zwork, &lzwork, &INFO );
  // G = V'BV
  ZUNMQR( "R", "N", &M, &n, &M, V, &n, (MKL_Complex16*)dwork,
          G, &n, zwork, &lzwork, &INFO ); 

  //---------------------------FIND-EIGENVALUES-OF-G--------------------------------
  il = 1; iu = blk;
  ZHEEVR( "N", "I", "U", &M, G, &n, &vl, &vu, &il, &iu, &tol, 
          &notneeded_m, TeValues, TeVectors, &M, isuppz, 
          zwork, &lzwork, dwork, &ldwork, iwork, &liwork, &INFO ); 
  // Eigenvalues stored in TeValues.
  DCOPY( &blk, TeValues, &iONE, ritzv, &iONE); // Copy the eigenvalues to output ritzv.
  //--------------------------------------------------------------------------------

  delete[] V;
  delete[] dwork;
  delete[] iwork;

  // depending on ctrl, may be NULL (no manual check needed - delete[] checks it)
  delete[] zwork;
  delete[] G;
  delete[] TeVectors;

  delete[] TeValues;
  delete[] isuppz;

  delete[] d;
  delete[] e;

  cout << "LANCZOS done." << endl;

  return;
}
