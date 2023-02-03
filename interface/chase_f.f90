!> @defgroup chasc-f ChASE F Interface
!>   @brief: this module provides a Fortran interface of ChASE  
!>  @{
!>
MODULE chase_diag
    ! non-MPI
    INTERFACE
        SUBROUTINE dchase_init(n, nev, nex, h, v, ritzv) bind( c, name = 'dchase_init_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex
            REAL(c_double)      :: h(n, *), v(n, *)
            REAL(c_double)      :: ritzv(*)

        END SUBROUTINE dchase_init    
    END INTERFACE

    INTERFACE
        SUBROUTINE dchase_finalize() bind( c, name = 'dchase_finalize_' )
            USE, INTRINSIC :: iso_c_binding

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
        SUBROUTINE schase_init(n, nev, nex, h, v, ritzv) bind( c, name = 'schase_init_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex
            REAL(c_float)      :: h(n, *), v(n, *)
            REAL(c_float)      :: ritzv(*)

        END SUBROUTINE schase_init    
    END INTERFACE

    INTERFACE
        SUBROUTINE schase_finalize() bind( c, name = 'schase_finalize_' )
            USE, INTRINSIC :: iso_c_binding

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
        SUBROUTINE cchase_init(n, nev, nex, h, v, ritzv) bind( c, name = 'cchase_init_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex
            COMPLEX(c_float_complex)      :: h(n, *), v(n, *)
            REAL(c_float)      :: ritzv(*)

        END SUBROUTINE cchase_init    
    END INTERFACE

    INTERFACE
        SUBROUTINE cchase_finalize() bind( c, name = 'cchase_finalize_' )
            USE, INTRINSIC :: iso_c_binding

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
        SUBROUTINE zchase_init(n, nev, nex, h, v, ritzv) bind( c, name = 'zchase_init_' )
            USE, INTRINSIC :: iso_c_binding
            INTEGER(c_int)      :: n, nev, nex
            COMPLEX(c_double_complex)      :: h(n, *), v(n, *)
            REAL(c_double)      :: ritzv(*)

        END SUBROUTINE zchase_init    
    END INTERFACE

    INTERFACE
        SUBROUTINE zchase_finalize() bind( c, name = 'zchase_finalize_' )
            USE, INTRINSIC :: iso_c_binding

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

END MODULE chase_diag
!> @} end of chasc-c


