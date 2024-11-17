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


    INTERFACE   
        SUBROUTINE pdchase_init(nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init) &
            bind( c, name = 'pdchase_init_f_' )
      !> Initialization of distributed-memory ChASE with real scalar in double precison.
      !> The matrix to be diagonalized is already in block-block distribution.
      !> It is linked to distributed multi-GPU ChASE when CUDA is detected.
      !>    
      !>
      !> @param[in] nn global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size      
      !> @param[in] h pointer to the matrix to be diagonalized. `h` is a block-block distribution of global matrix of size `mxn`, its leading dimension is `ldh`
      !> @param[in] ldh leading dimension of `h` on each MPI process
      !> @param[in] m max row number of local matrix `h` on each MPI process
      !> @param[in] n max column number of local matrix `h` on each MPI process
      !> @param[in,out] v `(mx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors. `v` is only partially distributed within column communicator. It is reduandant among different column communicator.
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in] dim0 row number of 2D MPI grid
      !> @param[in] dim1 column number of 2D MPI grid      
      !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`
      !> @param[in] fcomm the working MPI-Fortran communicator      
      !> @param[in,out] init a flag to indicate if ChASE has been initialized                 
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, m, n, ldh, dim0, dim1, fcomm, init
            REAL(c_double)      :: h(*), v(*)
            REAL(c_double)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major
        END SUBROUTINE pdchase_init
    END INTERFACE

    INTERFACE   
        SUBROUTINE pdchase_init_blockcyclic(nn, nev, nex, mbsize, nbsize, h, ldh, v, ritzv, dim0, dim1, &
            grid_major, irsrc, icsrc, fcomm, init) bind( c, name = 'pdchase_init_blockcyclic_f_' )
      !> Initialization of distributed-memory ChASE with real scalar in double precison.
      !> The matrix to be diagonalized is already in block-cyclic distribution.
      !> It is linked to distributed multi-GPU ChASE when CUDA is detected.
      !>    
      !>
      !> @param[in] nn global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size   
      !> @param[in] mbsize block size for the block-cyclic distribution for the rows of global matrix
      !> @param[in] nbsize block size for the block-cyclic distribution for the cloumns of global matrix
      !> @param[in] h pointer to the matrix to be diagonalized. `h` is a block-cyclic distribution of global matrix of size `mxn`, its leading dimension is `ldh`
      !> @param[in] ldh leading dimension of `h` on each MPI process
      !> @param[in,out] v `(mx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors. `v` is only partially distributed within column communicator in 1D block-cyclic distribution with a same block factor `mbsize`. It is reduandant among different column communicator.
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in] dim0 row number of 2D MPI grid
      !> @param[in] dim1 column number of 2D MPI grid      
      !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`
      !> @param[in] irsrc process row over which the first row of the global matrix `h` is distributed.
      !> @param[in] icsrc process column over which the first column of the global matrix `h` is distributed.      
      !> @param[in] fcomm the working MPI-Fortran communicator      
      !> @param[in,out] init a flag to indicate if ChASE has been initialized              
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, mbsize, nbsize, ldh, dim0, dim1, irsrc, icsrc, fcomm, init
            REAL(c_double)      :: h(*), v(*)
            REAL(c_double)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major

        END SUBROUTINE pdchase_init_blockcyclic
    END INTERFACE

    INTERFACE   
        SUBROUTINE pdchase_finalize(flag) bind( c, name = 'pdchase_finalize_' )
      !> Finalize distributed-memory ChASE with real scalar in double precison.
      !>    
      !>
      !> @param[in,out] flag A flag to indicate if ChASE has been cleared up            
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE pdchase_finalize
    END INTERFACE

    INTERFACE   
        SUBROUTINE pdchase(deg, tol, mode, opt, qr) bind( c, name = 'pdchase_' )
      !> Solve the eigenvalue by the previously constructed distributed-memory ChASE (real scalar in double precision).
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

        END SUBROUTINE pdchase    
    END INTERFACE


    INTERFACE   
        SUBROUTINE pschase_init(nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init) &
            bind( c, name = 'pschase_init_f_' )
      !> Initialization of distributed-memory ChASE with real scalar in single precison.
      !> The matrix to be diagonalized is already in block-block distribution.
      !> It is linked to distributed multi-GPU ChASE when CUDA is detected.
      !>    
      !>
      !> @param[in] nn global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size      
      !> @param[in] h pointer to the matrix to be diagonalized. `h` is a block-block distribution of global matrix of size `mxn`, its leading dimension is `ldh`
      !> @param[in] ldh leading dimension of `h` on each MPI process
      !> @param[in] m max row number of local matrix `h` on each MPI process
      !> @param[in] n max column number of local matrix `h` on each MPI process
      !> @param[in,out] v `(mx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors. `v` is only partially distributed within column communicator. It is reduandant among different column communicator.
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in] dim0 row number of 2D MPI grid
      !> @param[in] dim1 column number of 2D MPI grid      
      !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`
      !> @param[in] fcomm the working MPI-Fortran communicator      
      !> @param[in,out] init a flag to indicate if ChASE has been initialized              
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, m, n, ldh, dim0, dim1, fcomm, init
            REAL(c_float)      :: h(*), v(*)
            REAL(c_float)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major
        END SUBROUTINE pschase_init
    END INTERFACE
    
    INTERFACE
        SUBROUTINE pschase_init_blockcyclic(nn, nev, nex, mbsize, nbsize, h, ldh, v, ritzv, dim0, dim1, &
            grid_major, irsrc, icsrc, fcomm, init) bind( c, name = 'pschase_init_blockcyclic_f_' )
      !> Initialization of distributed-memory ChASE with real scalar in single precison.
      !> The matrix to be diagonalized is already in block-cyclic distribution.
      !> It is linked to distributed multi-GPU ChASE when CUDA is detected.
      !>    
      !>
      !> @param[in] nn global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size   
      !> @param[in] mbsize block size for the block-cyclic distribution for the rows of global matrix
      !> @param[in] nbsize block size for the block-cyclic distribution for the cloumns of global matrix
      !> @param[in] h pointer to the matrix to be diagonalized. `h` is a block-cyclic distribution of global matrix of size `mxn`, its leading dimension is `ldh`
      !> @param[in] ldh leading dimension of `h` on each MPI process
      !> @param[in,out] v `(mx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors. `v` is only partially distributed within column communicator in 1D block-cyclic distribution with a same block factor `mbsize`. It is reduandant among different column communicator.
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in] dim0 row number of 2D MPI grid
      !> @param[in] dim1 column number of 2D MPI grid      
      !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`
      !> @param[in] irsrc process row over which the first row of the global matrix `h` is distributed.
      !> @param[in] icsrc process column over which the first column of the global matrix `h` is distributed.      
      !> @param[in] fcomm the working MPI-Fortran communicator      
      !> @param[in,out] init a flag to indicate if ChASE has been initialized              
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, mbsize, nbsize, ldh, dim0, dim1, irsrc, icsrc, fcomm, init
            REAL(c_float)      :: h(*), v(*)
            REAL(c_float)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major

        END SUBROUTINE pschase_init_blockcyclic
    END INTERFACE

    INTERFACE  
        SUBROUTINE pschase_finalize(flag) bind( c, name = 'pschase_finalize_' )
      !> Finalize distributed-memory ChASE with real scalar in single precison.
      !>    
      !>
      !> @param[in,out] flag A flag to indicate if ChASE has been cleared up             
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE pschase_finalize
    END INTERFACE

    INTERFACE   
        SUBROUTINE pschase(deg, tol, mode, opt, qr) bind( c, name = 'pschase_' )
      !> Solve the eigenvalue by the previously constructed distributed-memory ChASE (real scalar in single precision).
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

        END SUBROUTINE pschase    
    END INTERFACE

    INTERFACE     
        SUBROUTINE pzchase_init(nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init) &
            bind( c, name = 'pzchase_init_f_' )
      !> Initialization of distributed-memory ChASE with complex scalar in double precison.
      !> The matrix to be diagonalized is already in block-block distribution.
      !> It is linked to distributed multi-GPU ChASE when CUDA is detected.
      !>    
      !>
      !> @param[in] nn global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size      
      !> @param[in] h pointer to the matrix to be diagonalized. `h` is a block-block distribution of global matrix of size `mxn`, its leading dimension is `ldh`
      !> @param[in] ldh leading dimension of `h` on each MPI process
      !> @param[in] m max row number of local matrix `h` on each MPI process
      !> @param[in] n max column number of local matrix `h` on each MPI process
      !> @param[in,out] v `(mx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors. `v` is only partially distributed within column communicator. It is reduandant among different column communicator.
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in] dim0 row number of 2D MPI grid
      !> @param[in] dim1 column number of 2D MPI grid      
      !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`
      !> @param[in] fcomm the working MPI-Fortran communicator      
      !> @param[in,out] init a flag to indicate if ChASE has been initialized              
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, m, n, ldh, dim0, dim1, fcomm, init
            COMPLEX(c_double_complex)      :: h(*), v(*)
            REAL(c_double)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major
        END SUBROUTINE pzchase_init
    END INTERFACE

    INTERFACE
        SUBROUTINE pzchase_init_blockcyclic(nn, nev, nex, mbsize, nbsize, h, ldh, v, ritzv, dim0, dim1, &
            grid_major, irsrc, icsrc, fcomm, init) bind( c, name = 'pzchase_init_blockcyclic_f_' )
      !> Initialization of distributed-memory ChASE with complex scalar in double precison.
      !> The matrix to be diagonalized is already in block-cyclic distribution.
      !> It is linked to distributed multi-GPU ChASE when CUDA is detected.
      !>    
      !>
      !> @param[in] nn global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size   
      !> @param[in] mbsize block size for the block-cyclic distribution for the rows of global matrix
      !> @param[in] nbsize block size for the block-cyclic distribution for the cloumns of global matrix
      !> @param[in] h pointer to the matrix to be diagonalized. `h` is a block-cyclic distribution of global matrix of size `mxn`, its leading dimension is `ldh`
      !> @param[in] ldh leading dimension of `h` on each MPI process
      !> @param[in,out] v `(mx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors. `v` is only partially distributed within column communicator in 1D block-cyclic distribution with a same block factor `mbsize`. It is reduandant among different column communicator.
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in] dim0 row number of 2D MPI grid
      !> @param[in] dim1 column number of 2D MPI grid      
      !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`
      !> @param[in] irsrc process row over which the first row of the global matrix `h` is distributed.
      !> @param[in] icsrc process column over which the first column of the global matrix `h` is distributed.      
      !> @param[in] fcomm the working MPI-Fortran communicator      
      !> @param[in,out] init a flag to indicate if ChASE has been initialized              
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, mbsize, nbsize, ldh, dim0, dim1, irsrc, icsrc, fcomm, init
            COMPLEX(c_double_complex)      :: h(*), v(*)
            REAL(c_double)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major

        END SUBROUTINE pzchase_init_blockcyclic
    END INTERFACE
    
    INTERFACE     
        SUBROUTINE pzchase_finalize(flag) bind( c, name = 'pzchase_finalize_' )
      !> Finalize distributed-memory ChASE with complex scalar in double precison.
      !>    
      !>
      !> @param[in,out] flag A flag to indicate if ChASE has been cleared up              
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE pzchase_finalize
    END INTERFACE

    INTERFACE     
        SUBROUTINE pzchase(deg, tol, mode, opt, qr) bind( c, name = 'pzchase_' )
      !> Solve the eigenvalue by the previously constructed distributed-memory ChASE (complex scalar in double precision).
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

        END SUBROUTINE pzchase    
    END INTERFACE

    INTERFACE 
        SUBROUTINE pcchase_init(nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init) &
            bind( c, name = 'pcchase_init_f_' )
      !> Initialization of distributed-memory ChASE with complex scalar in single precison.
      !> The matrix to be diagonalized is already in block-block distribution.
      !> It is linked to distributed multi-GPU ChASE when CUDA is detected.
      !>    
      !>
      !> @param[in] nn global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size      
      !> @param[in] h pointer to the matrix to be diagonalized. `h` is a block-block distribution of global matrix of size `mxn`, its leading dimension is `ldh`
      !> @param[in] ldh leading dimension of `h` on each MPI process
      !> @param[in] m max row number of local matrix `h` on each MPI process
      !> @param[in] n max column number of local matrix `h` on each MPI process
      !> @param[in,out] v `(mx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors. `v` is only partially distributed within column communicator. It is reduandant among differennt column communicator.
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in] dim0 row number of 2D MPI grid
      !> @param[in] dim1 column number of 2D MPI grid      
      !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`
      !> @param[in] fcomm the working MPI-Fortran communicator      
      !> @param[in,out] init a flag to indicate if ChASE has been initialized                  
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, m, n, ldh, dim0, dim1, fcomm, init
            COMPLEX(c_float_complex)      :: h(*), v(*)
            REAL(c_double)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major
        END SUBROUTINE pcchase_init
    END INTERFACE

    INTERFACE
        SUBROUTINE pcchase_init_blockcyclic(nn, nev, nex, mbsize, nbsize, h, ldh, v, ritzv, dim0, dim1, &
            grid_major, irsrc, icsrc, fcomm, init) bind( c, name = 'pcchase_init_blockcyclic_f_' )
      !> Initialization of distributed-memory ChASE with complex scalar in single precison.
      !> The matrix to be diagonalized is already in block-cyclic distribution.
      !> It is linked to distributed multi-GPU ChASE when CUDA is detected.
      !>    
      !>
      !> @param[in] nn global matrix size of the matrix to be diagonalized  
      !> @param[in] nev number of desired eigenpairs
      !> @param[in] nex extra searching space size   
      !> @param[in] mbsize block size for the block-cyclic distribution for the rows of global matrix
      !> @param[in] nbsize block size for the block-cyclic distribution for the cloumns of global matrix
      !> @param[in] h pointer to the matrix to be diagonalized. `h` is a block-cyclic distribution of global matrix of size `mxn`, its leading dimension is `ldh`
      !> @param[in] ldh leading dimension of `h` on each MPI process
      !> @param[in,out] v `(mx(nev+nex))` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors. `v` is only partially distributed within column communicator in 1D block-cyclic distribution with a same block factor `mbsize`. It is reduandant among different column communicator.
      !> @param[in,out] ritzv an array of size `nev` which contains the desired eigenvalues
      !> @param[in] dim0 row number of 2D MPI grid
      !> @param[in] dim1 column number of 2D MPI grid      
      !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`
      !> @param[in] irsrc process row over which the first row of the global matrix `h` is distributed.
      !> @param[in] icsrc process column over which the first column of the global matrix `h` is distributed.      
      !> @param[in] fcomm the working MPI-Fortran communicator      
      !> @param[in,out] init a flag to indicate if ChASE has been initialized              
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, mbsize, nbsize, ldh, dim0, dim1, irsrc, icsrc, fcomm, init
            COMPLEX(c_float_complex)      :: h(*), v(*)
            REAL(c_double)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major

        END SUBROUTINE pcchase_init_blockcyclic
    END INTERFACE

    INTERFACE   
        SUBROUTINE pcchase_finalize(flag) bind( c, name = 'pcchase_finalize_' )
      !> Finalize distributed-memory ChASE with complex scalar in single precison.
      !>    
      !>
      !> @param[in,out] flag A flag to indicate if ChASE has been cleared up                
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE pcchase_finalize
    END INTERFACE

    INTERFACE       
        SUBROUTINE pcchase(deg, tol, mode, opt, qr) bind( c, name = 'pcchase_' )
      !> Solve the eigenvalue by the previously constructed distributed-memory ChASE (complex scalar in single precision).
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

        END SUBROUTINE pcchase    
    END INTERFACE


END MODULE chase_diag

