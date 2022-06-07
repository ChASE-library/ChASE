MODULE chase_diag
  ! non-MPI
  INTERFACE
    SUBROUTINE dchase( h, n, v, ritzv, nev, nex, deg, tol, mode, opt ) bind( c, name = 'dchase_' )
      USE, INTRINSIC :: iso_c_binding
      REAL(c_double)        :: h(n,*), v(n,*)
      INTEGER(c_int)                :: n, deg, nev, nex
      REAL(c_double)                :: ritzv(*), tol
      CHARACTER(len=1,kind=c_char)  :: mode, opt
    END SUBROUTINE dchase
  END INTERFACE

  INTERFACE
    SUBROUTINE schase( h, n, v, ritzv, nev, nex, deg, tol, mode, opt ) bind( c, name = 'schase_' )
      USE, INTRINSIC :: iso_c_binding
      REAL(c_float)        :: h(n,*), v(n,*)
      INTEGER(c_int)                :: n, deg, nev, nex
      REAL(c_double)                :: ritzv(*), tol
      CHARACTER(len=1,kind=c_char)  :: mode, opt
    END SUBROUTINE schase
  END INTERFACE

  INTERFACE
    SUBROUTINE cchase( h, n, v, ritzv, nev, nex, deg, tol, mode, opt ) bind( c, name = 'cchase_' )
      USE, INTRINSIC :: iso_c_binding
      COMPLEX(c_float_complex)     :: h(n,*), v(n,*)
      INTEGER(c_int)                :: n, deg, nev, nex
      REAL(c_double)                :: ritzv(*), tol
      CHARACTER(len=1,kind=c_char)  :: mode, opt
    END SUBROUTINE cchase
  END INTERFACE

  INTERFACE
    SUBROUTINE zchase( h, n, v, ritzv, nev, nex, deg, tol, mode, opt ) bind( c, name = 'zchase_' )
      USE, INTRINSIC :: iso_c_binding
      COMPLEX(c_double_complex)     :: h(n,*), v(n,*)
      INTEGER(c_int)                :: n, deg, nev, nex
      REAL(c_double)                :: ritzv(*), tol
      CHARACTER(len=1,kind=c_char)  :: mode, opt
    END SUBROUTINE zchase
  END INTERFACE

  ! MPI
  INTERFACE
     SUBROUTINE pdchase_init( mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc) & 
                     BIND( c, name = 'pdchase_init' )
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, irsrc, icsrc
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pdchase_init
  END INTERFACE

  INTERFACE
     SUBROUTINE pzchase_init( mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc) &
                     BIND( c, name = 'pzchase_init' )
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, irsrc, icsrc
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pzchase_init
  END INTERFACE

  INTERFACE
     SUBROUTINE pschase_init( mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc) &
                     BIND( c, name = 'pschase_init' )
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, irsrc, icsrc
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pschase_init
  END INTERFACE

  INTERFACE
     SUBROUTINE pcchase_init( mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, grid_major, irsrc, icsrc) &
                     BIND( c, name = 'pcchase_init' )
       USE, INTRINSIC                :: iso_c_binding
       INTEGER(c_int)                :: mpi_comm, n, mbsize, nbsize, nev, nex, dim0, dim1, irsrc, icsrc
       CHARACTER(len=1,kind=c_char)  :: grid_major
     END SUBROUTINE pcchase_init
  END INTERFACE

  INTERFACE
     SUBROUTINE pdchase(h, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pdchase_' )
       USE, INTRINSIC :: iso_c_binding
       REAL(c_double)        :: h(*), v(*)
       INTEGER(c_int)                :: deg
       REAL(c_double)                :: ritzv(*), tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pdchase
  END INTERFACE

  INTERFACE
     SUBROUTINE pzchase(h, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pzchase_' )
       USE, INTRINSIC :: iso_c_binding
       COMPLEX(c_double_complex)     :: h(*), v(*)
       INTEGER(c_int)                :: deg
       REAL(c_double)                :: ritzv(*), tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pzchase
  END INTERFACE

  INTERFACE
     SUBROUTINE pschase(h, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pschase_' )
       USE, INTRINSIC :: iso_c_binding
       REAL(c_float)                 :: h(*), v(*)
       INTEGER(c_int)                :: deg
       REAL(c_float)                 :: ritzv(*)
       REAL(c_double)                :: tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pschase
  END INTERFACE

  INTERFACE
     SUBROUTINE pcchase(h, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pcchase_' )
       USE, INTRINSIC :: iso_c_binding
       COMPLEX(c_float_complex)      :: h(*), v(*)
       INTEGER(c_int)                :: deg
       REAL(c_float)                 :: ritzv(*)
       REAL(c_double)                :: tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pcchase
  END INTERFACE

#if defined(HAS_GPU)
  INTERFACE
     SUBROUTINE pdchase_mgpu(h, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pdchase_mgpu_' )
       USE, INTRINSIC :: iso_c_binding
       REAL(c_double)        :: h(*), v(*)
       INTEGER(c_int)                :: deg
       REAL(c_double)                :: ritzv(*), tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pdchase_mgpu
  END INTERFACE

  INTERFACE
     SUBROUTINE pzchase_mgpu(h, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pzchase_mgpu_' )
       USE, INTRINSIC :: iso_c_binding
       COMPLEX(c_double_complex)     :: h(*), v(*)
       INTEGER(c_int)                :: deg
       REAL(c_double)                :: ritzv(*), tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pzchase_mgpu
  END INTERFACE

  INTERFACE
     SUBROUTINE pschase_mgpu(h, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pschase_mgpu_' )
       USE, INTRINSIC :: iso_c_binding
       REAL(c_float)                 :: h(*), v(*)
       INTEGER(c_int)                :: deg
       REAL(c_float)                 :: ritzv(*)
       REAL(c_double)                :: tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pschase_mgpu
  END INTERFACE

  INTERFACE
     SUBROUTINE pcchase_mgpu(h, v, ritzv, deg, tol, mode, opt ) BIND( c, name = 'pcchase_mgpu_' )
       USE, INTRINSIC :: iso_c_binding
       COMPLEX(c_float_complex)      :: h(*), v(*)
       INTEGER(c_int)                :: deg
       REAL(c_float)                 :: ritzv(*)
       REAL(c_double)                :: tol
       CHARACTER(len=1,kind=c_char)  :: mode, opt
     END SUBROUTINE pcchase_mgpu
  END INTERFACE  
#endif

END MODULE chase_diag
