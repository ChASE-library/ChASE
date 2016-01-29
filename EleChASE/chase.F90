module m_chase
  PRIVATE

  interface
     subroutine fl_el_initialize(n,hmat,smat,mpi_com) bind (c)
       use, intrinsic :: iso_c_binding
       integer(kind=c_int),value                              :: n,mpi_com
       complex(kind=c_double_complex),dimension(*),intent(in) :: hmat,smat
     end subroutine fl_el_initialize
  end interface

  interface
     subroutine fl_el_diagonalize(neig,nex,deg,tol,mode,opt) bind (c)
       use, intrinsic :: iso_c_binding
       integer(kind=c_int),value                        :: neig,nex,deg,mode,opt
       real(kind=c_double),value                        :: tol
     end subroutine fl_el_diagonalize
  end interface

  interface
     subroutine fl_el_eigenvalues(neig,eig) bind (c)
       use, intrinsic :: iso_c_binding
       integer(kind=c_int),value                     :: neig
       real(kind=c_double),dimension(*),intent(out)  :: eig
     end subroutine fl_el_eigenvalues
  end interface

  interface
     subroutine fl_el_eigenvectors(neig,eig,z) bind (c)
       use, intrinsic :: iso_c_binding
       integer(kind=c_int),value                              :: neig
       real(kind=c_double),dimension(*),intent(out)           :: eig
       complex(kind=c_double_complex),dimension(*),intent(out):: z
     end subroutine fl_el_eigenvectors
  end interface

  PUBLIC chase

CONTAINS

  SUBROUTINE chase(m,n,SUB_COMM,nex,deg,tol,mode,opt,a,b,z,eig,num)
    !
    !----------------------------------------------------
    !
    ! m ........ actual (=leading) dimension of full a & b matrices
    !            must be problem size, as input a, b  are one-dimensional
    !            and shall be redistributed to two-dimensional matrices
    !            actual (=leading) dimension of eigenvector z(,)
    ! n ........ number of columns of full (sub)matrix ( about n/np)
    ! SUB_COMM.. communicator for MPI
    ! a,b   .... packed (sub)matrices, here expanded to non-packed
    ! z,eig .... eigenvectors and values, output
    ! num ...... number of ev's searched (and found) on this node
    !            On input, overall number of ev's searched,
    !            On output, local number of ev's found
    ! nex ...... initial size of the search subspace = nex + num, set to 0.2 * nev
    ! deg ...... set to 10
    ! tol ...... acc setting, 1e-10
    ! mode ..... 1 for random starting vectors and Lanczos
    ! opt ...... 1 for OPT_SINGLE, i.e. use deg as initial degree
    !----------------------------------------------------
    !
    IMPLICIT NONE
    INTEGER, INTENT (IN)                  :: m,n
    INTEGER, INTENT (IN)                  :: SUB_COMM
    INTEGER, INTENT (IN)                  :: nex,deg,mode,opt
    REAL(8), INTENT (IN)                  :: tol
    INTEGER, INTENT (INOUT)               :: num
    REAL(8),    INTENT   (OUT)               :: eig(:)
    COMPLEX(8), ALLOCATABLE, INTENT (INOUT)  :: a(:),b(:)
    COMPLEX(8), INTENT   (OUT)               :: z(:,:)

    INTEGER ::neig

    !Initialize the matrices in elemental
    call fl_el_initialize(m,a,b,SUB_COMM)
    !Now a,b could be deallocated
    !call diagonalization
    neig=num
    print*, "Elemental: seeking",neig," eigenvalues"
    call fl_el_diagonalize(neig,nex,deg,tol,mode,opt)
    print*, "Elemental found ",neig," local eigenvalues"
    num=min(size(z,2),neig)
    call fl_el_eigenvectors(num,eig,z)


  END subroutine chase
END module m_chase
