PROGRAM main
use mpi
use chase_diag

integer rank, size, ierr, init, i, j, k, comm
integer m, n, xoff, yoff, xlen, ylen, x, y, x_g, y_g
integer dims(2)
integer nn, nev, nex, idx_max
real(8) :: tmp, PI
real(8) :: tol
complex(8) :: cv
integer :: deg
character        :: mode, opt, major
complex(8),  allocatable :: h(:,:), v(:,:), hh(:, :)
real(8), allocatable :: lambda(:)

call mpi_init(ierr)
call mpi_comm_size(MPI_COMM_WORLD, size, ierr)
call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

nn = 1001
nev = 100
nex = 40
idx_max = 5

comm = MPI_COMM_WORLD
deg = 20
tol = 1e-10
mode = 'R'
opt = 'S'
major = 'C'

call mpi_dims_create(size, 2, dims, ierr) 

if( mod(nn, dims(1)) == 0) then
	m = nn /  dims(1);
else
	m = min(nn, nn / dims(1) + 1);
end if

if( mod(nn, dims(2)) == 0) then
	n = nn /  dims(2);
else
	n = min(nn, nn / dims(2) + 1);
end if

xoff = mod(rank, dims(1)) * m
yoff = mod(rank, dims(2)) * n

xlen = m;
ylen = n;

print *, xlen, ylen, xoff, yoff, rank
if(rank == 0) then
	print *, "ChASE Fortran example driver"
end if

allocate(h(m, n))
allocate(hh(nn, nn))
allocate(v(m, nev + nex))
allocate(lambda(nev + nex))

call pzchase_init(nn, nev, nex, m, n, h, m, v, lambda, dims(1), dims(2), major, comm, init)

!Generate Clement matrix in distributed manner
do x = 1, xlen
	do y = 1, ylen
	    x_g = xoff + x
	    y_g = yoff + y
	    if(x_g == y_g + 1) then
	       tmp = real(y_g * (nn + 1 - y_g))
	       h(x, y) = complex(sqrt(tmp), 0)
	    end if
	    if(y_g == x_g + 1) then
	       tmp = real(x_g * (nn + 1 - x_g))
	       h(x, y) = complex(sqrt(tmp), 0)
	    end if
	end do 
end do

call pzchase(deg, tol, mode, opt)

call pzchase_finalize(init)

call mpi_finalize(ierr)


END PROGRAM