PROGRAM main
use mpi
use chase_diag

integer rank, size, ierr, init
integer N, nev, nex, idx_max
real(8) :: perturb, tmp
real(8) :: tol
integer :: deg
character        :: mode, opt
complex(8),  allocatable :: h(:,:), v(:,:)
real(8), allocatable :: lambda(:)


call mpi_init(ierr)
call mpi_comm_size(MPI_COMM_WORLD, size, ierr)
call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

print *, 'Hello World from process: ', rank, 'of ', ierr

N = 1001
nev = 100
nex = 40
idx_max = 5
perturb = 1e-4

deg = 20
tol = 1e-10
mode = 'R'
opt = 'S'

if(rank == 0) then
	print *, "ChASE Fortran example driver"
end if

allocate(h(N, N)) 
allocate(v(N, nev+nex))
allocate(lambda(nev+nex))

call zchase_init(N, nev, nex, h, v, lambda)

! Generate Clement matrix
do i = 1, N
	h(i, i) = complex(0,0)
    do j = 1, N
    	if (i .ne. N - 1) then
  			tmp = real(i * (N + 1 - i))
    		h(i+1, i) = complex(sqrt(tmp), 0)
    	end if

    	if (i .ne. N - 1) then
  			tmp = real(i * (N + 1 - i))
    		h(i, i + 1) = complex(sqrt(tmp), 0)
    	end if
    end do
end do

call zchase(deg, tol, mode, opt)

call zchase_finalize()
call mpi_finalize(ierr)
END PROGRAM