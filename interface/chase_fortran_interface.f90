MODULE chase_diag
    ! non-MPI
    INTERFACE
        SUBROUTINE dchase_init(n, nev, nex, h, ldh, v, ritzv, init) bind( c, name = 'dchase_init_' )
      !> Initialization of shared-memory ChASE with real scalar in double precison.
      !> It is linked to single-GPU ChASE when CUDA is detected.
      !>    
      !>
      !> @param[in] n global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size      
      !> @param[in] h pointer to the matrix to be diagonalized
      !> @param[in] ldh a leading dimension of h
      !> @param[in,out] v `(nx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in,out] init a flag to indicate if ChASE has been initialized
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex, init, ldh
            REAL(c_double)      :: h(n, *), v(n, *)
            REAL(c_double)      :: ritzv(*)

        END SUBROUTINE dchase_init    
    END INTERFACE

    INTERFACE
      !> Finalize shared-memory ChASE with real scalar in double precison.
      !>    
      !>            
        SUBROUTINE dchase_finalize(flag) bind( c, name = 'dchase_finalize_' )
            USE, INTRINSIC :: iso_c_binding
      !> @param[in,out] flag A flag to indicate if ChASE has been cleared up              
            INTEGER(c_int)      :: flag

        END SUBROUTINE dchase_finalize    
    END INTERFACE

    INTERFACE 
        SUBROUTINE dchase(deg, tol, mode, opt, qr) bind( c, name = 'dchase_' )
      !> Solve the eigenvalue by the previously constructed shared-memory ChASE (real scalar in double precision).
      !> The buffer of matrix to be diagonalized, of ritz pairs have provided during the initialization of solver.
      !>    
      !>
      !> @param[in] deg initial degree of Cheyshev polynomial filter
      !> @param[in] tol desired absolute tolerance of computed eigenpairs
      !> @param[in] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.  
      !> @param[in] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.         
      !> @param[in] qr determining if flexible CholeskyQR, if `qr=C` use, otherwise, no use.                       
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_double)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt, qr

        END SUBROUTINE dchase    
    END INTERFACE

    INTERFACE
        SUBROUTINE schase_init(n, nev, nex, h, ldh, v, ritzv, init) bind( c, name = 'schase_init_' )
      !> Initialization of shared-memory ChASE with real scalar in single precison.
      !> It is linked to single-GPU ChASE when CUDA is detected.
      !>    
      !>
      !> @param[in] n global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size      
      !> @param[in] h pointer to the matrix to be diagonalized
      !> @param[in] ldh a leading dimension of h      
      !> @param[in,out] v `(nx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in,out] init a flag to indicate if ChASE has been initialized            
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex, init, ldh
            REAL(c_float)      :: h(n, *), v(n, *)
            REAL(c_float)      :: ritzv(*)

        END SUBROUTINE schase_init    
    END INTERFACE

    INTERFACE      
        SUBROUTINE schase_finalize(flag) bind( c, name = 'schase_finalize_' )
      !> Finalize shared-memory ChASE with real scalar in single precison.
      !>    
      !>
      !> @param[in,out] flag A flag to indicate if ChASE has been cleared up         
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE schase_finalize    
    END INTERFACE

    INTERFACE     
        SUBROUTINE schase(deg, tol, mode, opt, qr) bind( c, name = 'schase_' )
      !> Solve the eigenvalue by the previously constructed shared-memory ChASE (real scalar in single precision).
      !> The buffer of matrix to be diagonalized, of ritz pairs have provided during the initialization of solver.
      !>    
      !>
      !> @param[in] deg initial degree of Cheyshev polynomial filter
      !> @param[in] tol desired absolute tolerance of computed eigenpairs
      !> @param[in] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.  
      !> @param[in] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.         
      !> @param[in] qr determining if flexible CholeskyQR, if `qr=C` use, otherwise, no use.                                   
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_float)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt, qr

        END SUBROUTINE schase    
    END INTERFACE

    INTERFACE
        SUBROUTINE cchase_init(n, nev, nex, h, ldh, v, ritzv, init) bind( c, name = 'cchase_init_' )
      !> Initialization of shared-memory ChASE with complex scalar in single precison.
      !> It is linked to single-GPU ChASE when CUDA is detected.      
      !>    
      !>
      !> @param[in] n global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size      
      !> @param[in] h pointer to the matrix to be diagonalized
      !> @param[in] ldh a leading dimension of h      
      !> @param[in,out] v `(nx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in,out] init a flag to indicate if ChASE has been initialized            
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex, init, ldh
            COMPLEX(c_float_complex)      :: h(n, *), v(n, *)
            REAL(c_float)      :: ritzv(*)

        END SUBROUTINE cchase_init    
    END INTERFACE

    INTERFACE   
        SUBROUTINE cchase_finalize(flag) bind( c, name = 'cchase_finalize_' )
      !> Finalize shared-memory ChASE with complex scalar in single precison.
      !>    
      !>
      !> @param[in,out] flag a flag to indicate if ChASE has been cleared up            
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE cchase_finalize    
    END INTERFACE

    INTERFACE     
        SUBROUTINE cchase(deg, tol, mode, opt, qr) bind( c, name = 'cchase_' )
      !> Solve the eigenvalue by the previously constructed shared-memory ChASE (complex scalar in single precision).
      !> The buffer of matrix to be diagonalized, of ritz pairs have provided during the initialization of solver.
      !>    
      !>
      !> @param[in] deg initial degree of Cheyshev polynomial filter
      !> @param[in] tol desired absolute tolerance of computed eigenpairs
      !> @param[in] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.  
      !> @param[in] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.   
      !> @param[in] qr determining if flexible CholeskyQR, if `qr=C` use, otherwise, no use.                                   
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_float)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt, qr

        END SUBROUTINE cchase    
    END INTERFACE


    INTERFACE  
        SUBROUTINE zchase_init(n, nev, nex, h, ldh, v, ritzv, init) bind( c, name = 'zchase_init_' )
      !> Initialization of shard-memory ChASE with complex scalar in double precison.
      !> It is linked to single-GPU ChASE when CUDA is detected.      
      !>    
      !>
      !> @param[in] n global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size      
      !> @param[in] h pointer to the matrix to be diagonalized
      !> @param[in] ldh a leading dimension of h      
      !> @param[in,out] v `(nx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in,out] init a flag to indicate if ChASE has been initialized         
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex, init, ldh
            COMPLEX(c_double_complex)      :: h(n, *), v(n, *)
            REAL(c_double)      :: ritzv(*)

        END SUBROUTINE zchase_init    
    END INTERFACE

    INTERFACE       
        SUBROUTINE zchase_finalize(flag) bind( c, name = 'zchase_finalize_' )
      !> Finalize shared-memory ChASE with complex scalar in double precison.
      !>    
      !>
      !> @param[in,out] flag a flag to indicate if ChASE has been cleared up        
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE zchase_finalize    
    END INTERFACE

    INTERFACE     
        SUBROUTINE zchase(deg, tol, mode, opt, qr) bind( c, name = 'zchase_' )
      !> Solve the eigenvalue by the previously constructed shared-memory ChASE (complex scalar in double precision).
      !> The buffer of matrix to be diagonalized, of ritz pairs have provided during the initialization of solver.
      !>    
      !>
      !> @param[in] deg initial degree of Cheyshev polynomial filter
      !> @param[in] tol desired absolute tolerance of computed eigenpairs
      !> @param[in] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.  
      !> @param[in] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.  
      !> @param[in] qr determining if flexible CholeskyQR, if `qr=C` use, otherwise, no use.                                     
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_double)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt, qr

        END SUBROUTINE zchase    
    END INTERFACE
END MODULE chase_diag

