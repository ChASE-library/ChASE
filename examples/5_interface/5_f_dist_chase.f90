PROGRAM main
use mpi
use chase_diag

integer rank, size, ierr, init, i, j, k, comm
integer m, n, xoff, yoff, xlen, ylen, x, y
integer dims(2)
integer nn, nev, nex, idx_max
real(8) :: perturb, tmp, PI
real(8) :: tol
complex(8) :: cv
integer :: deg
character        :: mode, opt, major
complex(8),  allocatable :: h(:,:), v(:,:), hh(:, :)
real(8), allocatable :: lambda(:)

call mpi_init(ierr)
call mpi_comm_size(MPI_COMM_WORLD, size, ierr)
call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

if(size .ne. 2) then
	print *, "This example is only desigend for MPI comm size = 2"

	return
end if

nn = 1001
nev = 100
nex = 40
idx_max = 5
perturb = 1e-4

comm = MPI_COMM_WORLD
deg = 20
tol = 1e-10
mode = 'R'
opt = 'S'
major = 'C'
dims(1) = 2
dims(2) = 1

m = 501
n = nn

if(rank == 0) then
	xoff = 0
else
	xoff = 501
end if

yoff = 0

xlen = m
ylen = n

if(rank == 0) then
	print *, "ChASE Fortran example driver"
end if

allocate(h(m, n))
allocate(hh(nn, nn))
allocate(v(m, nev + nex))
allocate(lambda(nev + nex))

call pzchase_init(nn, nev, nex, m, n, h, m, v, lambda, dims(1), dims(2), major, comm, init)

! Generate Clement matrix
do i = 1, N
	hh(i, i) = complex(0,0)
    do j = 1, N
    	if (i .ne. N - 1) then
  			tmp = real(i * (N + 1 - i))
    		hh(i+1, i) = complex(sqrt(tmp), 0)
    	end if

    	if (i .ne. N - 1) then
  			tmp = real(i * (N + 1 - i))
    		hh(i, i + 1) = complex(sqrt(tmp), 0)
    	end if
    end do
end do


do x = 1, xlen
	do y = 1, ylen
		h(x, y) = hh(xoff + x, yoff + y)
	end do 
end do

call pzchase(deg, tol, mode, opt)

call pzchase_finalize(init)

call mpi_finalize(ierr)


END PROGRAM