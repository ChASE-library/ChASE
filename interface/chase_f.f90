MODULE chase_diag
  ! non-MPI
  INTERFACE
    SUBROUTINE rchase( h, n, v, ritzv, nev, nex, deg, tol, mode, opt ) bind( c, name = 'dchase_' )
      USE, INTRINSIC :: iso_c_binding
      REAL(c_double)        :: h(n,*), v(n,*)
      INTEGER(c_int)                :: n, deg, nev, nex
      REAL(c_double)                :: ritzv(*), tol
      CHARACTER(len=1,kind=c_char)  :: mode, opt
    END SUBROUTINE rchase
  END INTERFACE

  INTERFACE
    SUBROUTINE cchase( h, n, v, ritzv, nev, nex, deg, tol, mode, opt ) bind( c, name = 'zchase_' )
      USE, INTRINSIC :: iso_c_binding
      COMPLEX(c_double_complex)     :: h(n,*), v(n,*)
      INTEGER(c_int)                :: n, deg, nev, nex
      REAL(c_double)                :: ritzv(*), tol
      CHARACTER(len=1,kind=c_char)  :: mode, opt
    END SUBROUTINE cchase
  END INTERFACE

  ! MPI
  INTERFACE
     SUBROUTINE prchase_init( mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc) & 
                     BIND( c, name = 'dchase_init' )
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, irsrc, icsrc
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE prchase_init
  END INTERFACE

  INTERFACE
     SUBROUTINE pcchase_init( mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc) &
                     BIND( c, name = 'zchase_init' )
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, irsrc, icsrc
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pcchase_init
  END INTERFACE

  INTERFACE
     SUBROUTINE prchase(h, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'dchase_solve' )
       USE, INTRINSIC :: iso_c_binding
       REAL(c_double)        :: h(*), v(*)
       INTEGER(c_int)                :: deg
       REAL(c_double)                :: ritzv(*), tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE prchase
  END INTERFACE

  INTERFACE
     SUBROUTINE pcchase(h, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'zchase_solve' )
       USE, INTRINSIC :: iso_c_binding
       COMPLEX(c_double_complex)     :: h(*), v(*)
       INTEGER(c_int)                :: deg
       REAL(c_double)                :: ritzv(*), tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pcchase
  END INTERFACE

END MODULE chase_diag
