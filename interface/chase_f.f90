!> @defgroup chasc-f ChASE F Interface
!>   @brief: this module provides a Fortran interface of ChASE  
!>  @{
!>
MODULE chase_diag
    ! non-MPI
    INTERFACE
        SUBROUTINE dchase_init(n, nev, nex, h, v, ritzv, init) bind( c, name = 'dchase_init_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex, init
            REAL(c_double)      :: h(n, *), v(n, *)
            REAL(c_double)      :: ritzv(*)

        END SUBROUTINE dchase_init    
    END INTERFACE

    INTERFACE
        SUBROUTINE dchase_finalize(flag) bind( c, name = 'dchase_finalize_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE dchase_finalize    
    END INTERFACE

    INTERFACE
        SUBROUTINE dchase(deg, tol, mode, opt) bind( c, name = 'dchase_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_double)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt

        END SUBROUTINE dchase    
    END INTERFACE

    INTERFACE
        SUBROUTINE schase_init(n, nev, nex, h, v, ritzv, init) bind( c, name = 'schase_init_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex, init
            REAL(c_float)      :: h(n, *), v(n, *)
            REAL(c_float)      :: ritzv(*)

        END SUBROUTINE schase_init    
    END INTERFACE

    INTERFACE
        SUBROUTINE schase_finalize(flag) bind( c, name = 'schase_finalize_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE schase_finalize    
    END INTERFACE

    INTERFACE
        SUBROUTINE schase(deg, tol, mode, opt) bind( c, name = 'schase_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_float)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt

        END SUBROUTINE schase    
    END INTERFACE

    INTERFACE
        SUBROUTINE cchase_init(n, nev, nex, h, v, ritzv, init) bind( c, name = 'cchase_init_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex, init
            COMPLEX(c_float_complex)      :: h(n, *), v(n, *)
            REAL(c_float)      :: ritzv(*)

        END SUBROUTINE cchase_init    
    END INTERFACE

    INTERFACE
        SUBROUTINE cchase_finalize(flag) bind( c, name = 'cchase_finalize_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE cchase_finalize    
    END INTERFACE

    INTERFACE
        SUBROUTINE cchase(deg, tol, mode, opt) bind( c, name = 'cchase_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_float)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt

        END SUBROUTINE cchase    
    END INTERFACE


    INTERFACE
        SUBROUTINE zchase_init(n, nev, nex, h, v, ritzv, init) bind( c, name = 'zchase_init_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex, init
            COMPLEX(c_double_complex)      :: h(n, *), v(n, *)
            REAL(c_double)      :: ritzv(*)

        END SUBROUTINE zchase_init    
    END INTERFACE

    INTERFACE
        SUBROUTINE zchase_finalize(flag) bind( c, name = 'zchase_finalize_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE zchase_finalize    
    END INTERFACE

    INTERFACE
        SUBROUTINE zchase(deg, tol, mode, opt) bind( c, name = 'zchase_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_double)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt

        END SUBROUTINE zchase    
    END INTERFACE


    INTERFACE
        SUBROUTINE pdchase_init(nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init) &
            bind( c, name = 'pdchase_init_f_' )
            
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
            
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, mbsize, nbsize, ldh, dim0, dim1, irsrc, icsrc, fcomm, init
            REAL(c_double)      :: h(*), v(*)
            REAL(c_double)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major

        END SUBROUTINE pdchase_init_blockcyclic
    END INTERFACE

    INTERFACE
        SUBROUTINE pdchase_finalize(flag) bind( c, name = 'pdchase_finalize_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE pdchase_finalize
    END INTERFACE

    INTERFACE
        SUBROUTINE pdchase(deg, tol, mode, opt) bind( c, name = 'pdchase_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_double)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt

        END SUBROUTINE pdchase    
    END INTERFACE


    INTERFACE
        SUBROUTINE pschase_init(nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init) &
            bind( c, name = 'pschase_init_f_' )
            
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
            
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, mbsize, nbsize, ldh, dim0, dim1, irsrc, icsrc, fcomm, init
            REAL(c_float)      :: h(*), v(*)
            REAL(c_float)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major

        END SUBROUTINE pschase_init_blockcyclic
    END INTERFACE
    
    INTERFACE
        SUBROUTINE pschase_finalize(flag) bind( c, name = 'pschase_finalize_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE pschase_finalize
    END INTERFACE

    INTERFACE
        SUBROUTINE pschase(deg, tol, mode, opt) bind( c, name = 'pschase_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_float)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt

        END SUBROUTINE pschase    
    END INTERFACE

    INTERFACE
        SUBROUTINE pzchase_init(nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init) &
            bind( c, name = 'pzchase_init_f_' )
            
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
            
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, mbsize, nbsize, ldh, dim0, dim1, irsrc, icsrc, fcomm, init
            COMPLEX(c_double_complex)      :: h(*), v(*)
            REAL(c_double)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major

        END SUBROUTINE pzchase_init_blockcyclic
    END INTERFACE

    INTERFACE
        SUBROUTINE pzchase_finalize(flag) bind( c, name = 'pzchase_finalize_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE pzchase_finalize
    END INTERFACE

    INTERFACE
        SUBROUTINE pzchase(deg, tol, mode, opt) bind( c, name = 'pzchase_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_double)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt

        END SUBROUTINE pzchase    
    END INTERFACE

    INTERFACE
        SUBROUTINE pcchase_init(nn, nev, nex, m, n, h, ldh, v, ritzv, dim0, dim1, grid_major, fcomm, init) &
            bind( c, name = 'pcchase_init_f_' )
            
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
            
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: nn, nev, nex, mbsize, nbsize, ldh, dim0, dim1, irsrc, icsrc, fcomm, init
            COMPLEX(c_float_complex)      :: h(*), v(*)
            REAL(c_double)      :: ritzv(*)
            CHARACTER(len=1,kind=c_char)  :: grid_major

        END SUBROUTINE pcchase_init_blockcyclic
    END INTERFACE

    INTERFACE
        SUBROUTINE pcchase_finalize(flag) bind( c, name = 'pcchase_finalize_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: flag

        END SUBROUTINE pcchase_finalize
    END INTERFACE

    INTERFACE
        SUBROUTINE pcchase(deg, tol, mode, opt) bind( c, name = 'pcchase_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: deg
            REAL(c_float)      :: tol
            CHARACTER(len=1,kind=c_char)  :: mode, opt

        END SUBROUTINE pcchase    
    END INTERFACE

END MODULE chase_diag
!> @} end of chasc-c


