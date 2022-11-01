!> @defgroup chasc-f ChASE F Interface
!>   @brief: this module provides a Fortran interface of ChASE  
!>  @{
!>
MODULE chase_diag
  ! non-MPI
  INTERFACE
  !> shard-memory version of ChASE with real scalar in double precison
  !>  
    SUBROUTINE dchase( h, n, v, ritzv, nev, nex, deg, tol, mode, opt ) bind( c, name = 'dchase_' )
  !>
  !> @param[in] h pointer to the matrix to be diagonalized
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] nev number of desired eigenpairs
  !> @param[int] nex extra searching space size
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.    
      USE, INTRINSIC :: iso_c_binding
      REAL(c_double)        :: h(n,*), v(n,*)
      INTEGER(c_int)                :: n, deg, nev, nex
      REAL(c_double)                :: ritzv(*), tol
      CHARACTER(len=1,kind=c_char)  :: mode, opt
    END SUBROUTINE dchase
  END INTERFACE

  INTERFACE
  !> shard-memory version of ChASE with real scalar in single precison
  !>
!>    
    SUBROUTINE schase( h, n, v, ritzv, nev, nex, deg, tol, mode, opt ) bind( c, name = 'schase_' )
  !> @param[in] h pointer to the matrix to be diagonalized
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] nev number of desired eigenpairs
  !> @param[int] nex extra searching space size
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.    
      USE, INTRINSIC :: iso_c_binding
      REAL(c_float)        :: h(n,*), v(n,*)
      INTEGER(c_int)                :: n, deg, nev, nex
      REAL(c_double)                :: ritzv(*), tol
      CHARACTER(len=1,kind=c_char)  :: mode, opt
    END SUBROUTINE schase
  END INTERFACE
  !> shard-memory version of ChASE with complex scalar in single precison
  !>
!>  
  INTERFACE
    SUBROUTINE cchase( h, n, v, ritzv, nev, nex, deg, tol, mode, opt ) bind( c, name = 'cchase_' )
  !> @param[in] h pointer to the matrix to be diagonalized
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] nev number of desired eigenpairs
  !> @param[int] nex extra searching space size
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.    
      USE, INTRINSIC :: iso_c_binding
      COMPLEX(c_float_complex)     :: h(n,*), v(n,*)
      INTEGER(c_int)                :: n, deg, nev, nex
      REAL(c_double)                :: ritzv(*), tol
      CHARACTER(len=1,kind=c_char)  :: mode, opt
    END SUBROUTINE cchase
  END INTERFACE
  !> shard-memory version of ChASE with complex scalar in double precison
  !>
!>  
  INTERFACE
    SUBROUTINE zchase( h, n, v, ritzv, nev, nex, deg, tol, mode, opt ) bind( c, name = 'zchase_' )
  !> @param[in] h pointer to the matrix to be diagonalized
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] nev number of desired eigenpairs
  !> @param[int] nex extra searching space size
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.    
      USE, INTRINSIC :: iso_c_binding
      COMPLEX(c_double_complex)     :: h(n,*), v(n,*)
      INTEGER(c_int)                :: n, deg, nev, nex
      REAL(c_double)                :: ritzv(*), tol
      CHARACTER(len=1,kind=c_char)  :: mode, opt
    END SUBROUTINE zchase
  END INTERFACE

  ! MPI
  INTERFACE
  !> an initialisation of environment for distributed ChASE for real scalar in double precision
  !>
  !>   
     SUBROUTINE pdchase_init( mpi_comm, n, nev, nex) &
                     BIND( c, name = 'pdchase_init' )
  !> A built-in mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size                     
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, nev, nex
     END SUBROUTINE pdchase_init
  END INTERFACE
  !> an initialisation of environment for distributed ChASE for complex scalar in double precision
  !>
  !>  
  INTERFACE
     SUBROUTINE pzchase_init( mpi_comm, n, nev, nex) &
                     BIND( c, name = 'pzchase_init' )
  !> A built-in mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size                     
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, nev, nex
     END SUBROUTINE pzchase_init
  END INTERFACE
  !> an initialisation of environment for distributed ChASE for complex scalar in single precision
  !>
  INTERFACE
     SUBROUTINE pcchase_init( mpi_comm, n, nev, nex) &
                     BIND( c, name = 'pcchase_init' )
  !> A built-in mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size
  !>                       
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, nev, nex
     END SUBROUTINE pcchase_init
  END INTERFACE
  !> an initialisation of environment for distributed ChASE for real scalar in single precision
  !>
  !>  
  INTERFACE
     SUBROUTINE pschase_init( mpi_comm, n, nev, nex) &
                     BIND( c, name = 'pschase_init' )
  !> A built-in mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size                     
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, nev, nex
     END SUBROUTINE pschase_init
  END INTERFACE
  !> an initialisation of environment for distributed ChASE for real scalar in double precision
  !>  
  INTERFACE
     SUBROUTINE pdchase_init_block( mpi_comm, n, nev, nex, m_, n_, dim0, dim1, grid_major) &
                     BIND( c, name = 'pdchase_init_block' )
  !> A mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE in Block distribution faison
  !> This mechanism is built with user provided MPI grid shape and maximum block in row/column direction
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size
  !> @param[in] m_ max row number of local matrix on each MPI process
  !> @param[in] n_ max column number of local matrix on each MPI process
  !> @param[in] dim0 row number of 2D MPI grid
  !> @param[in] dim1 column number of 2D MPI grid
  !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`. 
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, nev, nex, m_, n_, dim0, dim1
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pdchase_init_block
  END INTERFACE
  !> an initialisation of environment for distributed ChASE for complex scalar in double precision
  !>  
  INTERFACE
     SUBROUTINE pzchase_init_block( mpi_comm, n, nev, nex, m_, n_, dim0, dim1, grid_major) &
                     BIND( c, name = 'pzchase_init_block' )
  !> A mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE in Block distribution faison
  !> This mechanism is built with user provided MPI grid shape and maximum block in row/column direction
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size
  !> @param[in] m_ max row number of local matrix on each MPI process
  !> @param[in] n_ max column number of local matrix on each MPI process
  !> @param[in] dim0 row number of 2D MPI grid
  !> @param[in] dim1 column number of 2D MPI grid
  !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`.                      
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, nev, nex, m_, n_, dim0, dim1
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pzchase_init_block
  END INTERFACE
  !> an initialisation of environment for distributed ChASE for complex scalar in single precision
  !>  
  INTERFACE
     SUBROUTINE pcchase_init_block( mpi_comm, n, nev, nex, m_, n_, dim0, dim1, grid_major) &
                     BIND( c, name = 'pcchase_init_block' )
  !> A mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE in Block distribution faison
  !> This mechanism is built with user provided MPI grid shape and maximum block in row/column direction
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size
  !> @param[in] m_ max row number of local matrix on each MPI process
  !> @param[in] n_ max column number of local matrix on each MPI process
  !> @param[in] dim0 row number of 2D MPI grid
  !> @param[in] dim1 column number of 2D MPI grid
  !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`.                       
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, nev, nex, m_, n_, dim0, dim1
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pcchase_init_block
  END INTERFACE
  !> an initialisation of environment for distributed ChASE for real scalar in single precision
  !>  
  INTERFACE
     SUBROUTINE pschase_init_block( mpi_comm, n, nev, nex, m_, n_, dim0, dim1, grid_major) &
                     BIND( c, name = 'pschase_init_block' )
  !> A mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE in Block distribution faison
  !> This mechanism is built with user provided MPI grid shape and maximum block in row/column direction
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size
  !> @param[in] m_ max row number of local matrix on each MPI process
  !> @param[in] n_ max column number of local matrix on each MPI process
  !> @param[in] dim0 row number of 2D MPI grid
  !> @param[in] dim1 column number of 2D MPI grid
  !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`.                       
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, nev, nex, m_, n_, dim0, dim1
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pschase_init_block
  END INTERFACE
  !> an initialisation of environment for distributed ChASE for real scalar in double precision
  !>  
  INTERFACE
     SUBROUTINE pdchase_init_blockcyclic( mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc) & 
                     BIND( c, name = 'pdchase_init_blockcyclic' )
  !> A mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE in Block-Cylic distribution faison
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] mbsize block size for the block-cyclic distribution in the row direction
  !> @param[in] nbsize block size for the block-cyclic distribution in the column direction
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size
  !> @param[in] dim0 row number of 2D MPI grid
  !> @param[in] dim1 column number of 2D MPI grid
  !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`.       
  !> @param[in] irsrc Process row over which the first row of the global matrix `A` is distributed.
  !> @param[in] icsrc Process column over which the first column of the global matrix `A` is distributed.                 
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, irsrc, icsrc
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pdchase_init_blockcyclic
  END INTERFACE
  !> an initialisation of environment for distributed ChASE for complex scalar in double precision
  !>  
  INTERFACE
     SUBROUTINE pzchase_init_blockcyclic( mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc) &
                     BIND( c, name = 'pzchase_init_blockcyclic' )
  !> A mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE in Block-Cylic distribution faison
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] mbsize block size for the block-cyclic distribution in the row direction
  !> @param[in] nbsize block size for the block-cyclic distribution in the column direction
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size
  !> @param[in] dim0 row number of 2D MPI grid
  !> @param[in] dim1 column number of 2D MPI grid
  !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`.       
  !> @param[in] irsrc Process row over which the first row of the global matrix `A` is distributed.
  !> @param[in] icsrc Process column over which the first column of the global matrix `A` is distributed.                      
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, irsrc, icsrc
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pzchase_init_blockcyclic
  END INTERFACE
  !> an initialisation of environment for distributed ChASE for real scalar in single precision
  !>  
  INTERFACE
     SUBROUTINE pschase_init_blockcyclic( mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc) &
                     BIND( c, name = 'pschase_init_blockcyclic' )
  !> A mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE in Block-Cylic distribution faison
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] mbsize block size for the block-cyclic distribution in the row direction
  !> @param[in] nbsize block size for the block-cyclic distribution in the column direction
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size
  !> @param[in] dim0 row number of 2D MPI grid
  !> @param[in] dim1 column number of 2D MPI grid
  !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`.       
  !> @param[in] irsrc Process row over which the first row of the global matrix `A` is distributed.
  !> @param[in] icsrc Process column over which the first column of the global matrix `A` is distributed.                      
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, irsrc, icsrc
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pschase_init_blockcyclic
  END INTERFACE
  !> an initialisation of environment for distributed ChASE for complex scalar in single precision
  !>  
  INTERFACE
     SUBROUTINE pcchase_init_blockcyclic( mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc) &
                     BIND( c, name = 'pcchase_init_blockcyclic' )
  !> A mechanism is used to distributed the Hermitian/Symmetric matrices in ChASE in Block-Cylic distribution faison
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] mpi_comm the working MPI communicator
  !> @param[in] n global matrix size of the matrix to be diagonalized
  !> @param[in] mbsize block size for the block-cyclic distribution in the row direction
  !> @param[in] nbsize block size for the block-cyclic distribution in the column direction
  !> @param[in] nev number of desired eigenpairs
  !> @param[in] nex extra searching space size
  !> @param[in] dim0 row number of 2D MPI grid
  !> @param[in] dim1 column number of 2D MPI grid
  !> @param[in] grid_major major of 2D MPI grid. Row major: `grid_major=R`, column major: `grid_major=C`.       
  !> @param[in] irsrc Process row over which the first row of the global matrix `A` is distributed.
  !> @param[in] icsrc Process column over which the first column of the global matrix `A` is distributed.                   
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, irsrc, icsrc
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pcchase_init_blockcyclic
  END INTERFACE
  !> distributed CPU version ChASE for real scalar in double precision
  !> 
  INTERFACE
     SUBROUTINE pdchase(h, ldh, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pdchase_' )
  !> Compute the first nev eigenpairs by ChASE
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] h pointer to the local portion of the matrix to be diagonalized
  !> @param[in] ldh leading dimension of `h`
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.    
       USE, INTRINSIC :: iso_c_binding
       REAL(c_double)        :: h(*), v(*)
       INTEGER(c_int)                :: deg, ldh
       REAL(c_double)                :: ritzv(*), tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pdchase
  END INTERFACE
  !> distributed CPU version ChASE for complex scalar in double precision
  !> 
  INTERFACE
     SUBROUTINE pzchase(h, ldh, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pzchase_' )
  !> Compute the first nev eigenpairs by ChASE
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] h pointer to the local portion of the matrix to be diagonalized
  !> @param[in] ldh leading dimension of `h`
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.    
       USE, INTRINSIC :: iso_c_binding
       COMPLEX(c_double_complex)     :: h(*), v(*)
       INTEGER(c_int)                :: deg, ldh
       REAL(c_double)                :: ritzv(*), tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pzchase
  END INTERFACE
  !> distributed CPU version ChASE for real scalar in single precision
  !> 
  INTERFACE
     SUBROUTINE pschase(h, ldh, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pschase_' )
  !> Compute the first nev eigenpairs by ChASE
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] h pointer to the local portion of the matrix to be diagonalized
  !> @param[in] ldh leading dimension of `h`
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.    
       USE, INTRINSIC :: iso_c_binding
       REAL(c_float)                 :: h(*), v(*)
       INTEGER(c_int)                :: deg, ldh
       REAL(c_float)                 :: ritzv(*)
       REAL(c_double)                :: tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pschase
  END INTERFACE
  !> distributed CPU version ChASE for complex scalar in single precision
  !> 
  INTERFACE
     SUBROUTINE pcchase(h, ldh, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pcchase_' )
  !> Compute the first nev eigenpairs by ChASE
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] h pointer to the local portion of the matrix to be diagonalized
  !> @param[in] ldh leading dimension of `h`
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.         
       USE, INTRINSIC :: iso_c_binding
       COMPLEX(c_float_complex)      :: h(*), v(*)
       INTEGER(c_int)                :: deg, ldh
       REAL(c_float)                 :: ritzv(*)
       REAL(c_double)                :: tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pcchase
  END INTERFACE

#if defined(HAS_GPU)
  !> distributed multi-GPU version ChASE for real scalar in double precision
  !> 
  INTERFACE
     SUBROUTINE pdchase_mgpu(h, ldh, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pdchase_mgpu_' )
  !> Compute the first nev eigenpairs by ChASE
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] h pointer to the local portion of the matrix to be diagonalized
  !> @param[in] ldh leading dimension of `h`
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.         
       USE, INTRINSIC :: iso_c_binding
       REAL(c_double)        :: h(*), v(*)
       INTEGER(c_int)                :: deg, ldh
       REAL(c_double)                :: ritzv(*), tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pdchase_mgpu
  END INTERFACE
  !> distributed multi-GPU version ChASE for complex scalar in double precision
  !> 
  INTERFACE
     SUBROUTINE pzchase_mgpu(h, ldh, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pzchase_mgpu_' )
  !> Compute the first nev eigenpairs by ChASE
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] h pointer to the local portion of the matrix to be diagonalized
  !> @param[in] ldh leading dimension of `h`
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.         
       USE, INTRINSIC :: iso_c_binding
       COMPLEX(c_double_complex)     :: h(*), v(*)
       INTEGER(c_int)                :: deg, ldh
       REAL(c_double)                :: ritzv(*), tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pzchase_mgpu
  END INTERFACE
  !> distributed multi-GPU version ChASE for real scalar in single precision
  !> 
  INTERFACE
     SUBROUTINE pschase_mgpu(h, ldh, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pschase_mgpu_' )
  !> Compute the first nev eigenpairs by ChASE
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] h pointer to the local portion of the matrix to be diagonalized
  !> @param[in] ldh leading dimension of `h`
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.         
       USE, INTRINSIC :: iso_c_binding
       REAL(c_float)                 :: h(*), v(*)
       INTEGER(c_int)                :: deg, ldh
       REAL(c_float)                 :: ritzv(*)
       REAL(c_double)                :: tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pschase_mgpu
  END INTERFACE
  !> distributed multi-GPU version ChASE for complex scalar in single precision
  !> 
  INTERFACE
     SUBROUTINE pcchase_mgpu(h, ldh, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pcchase_mgpu_' )
  !> Compute the first nev eigenpairs by ChASE
  !> This mechanism is built with user provided MPI grid shape and blocksize of block-cyclic distribution in row/column direction
  !> @param[in] h pointer to the local portion of the matrix to be diagonalized
  !> @param[in] ldh leading dimension of `h`
  !> @param[inout] v `(Nxnev+nex)` matrix, input is the initial guess eigenvectors, and for output, the first `nev` columns are overwritten by the desired eigenvectors
  !> @param[out] ritzv an array of size `nev` which contains the desired eigenvalues
  !> @param[int] deg initial degree of Cheyshev polynomial filter
  !> @param[int] tol desired absolute tolerance of computed eigenpairs
  !> @param[int] mode for sequences of eigenproblems, if reusing the eigenpairs obtained from last system. If `mode = A`, reuse, otherwise, not.
  !> @param[int] opt determining if using internal optimization of Chebyshev polynomial degree. If `opt=S`, use, otherwise, no.         
       USE, INTRINSIC :: iso_c_binding
       COMPLEX(c_float_complex)      :: h(*), v(*)
       INTEGER(c_int)                :: deg, ldh
       REAL(c_float)                 :: ritzv(*)
       REAL(c_double)                :: tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pcchase_mgpu
  END INTERFACE  
#endif

END MODULE chase_diag
!> @} end of chasc-c