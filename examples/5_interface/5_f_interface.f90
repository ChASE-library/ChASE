PROGRAM main
use mpi
use chase_diag

integer rank, size, ierr

call MPI_INIT(ierr)
call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

print *, 'Hello World from process: ', rank, 'of ', ierr

call MPI_FINALIZE(ierr)
END PROGRAM