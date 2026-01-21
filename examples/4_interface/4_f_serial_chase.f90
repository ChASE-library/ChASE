! This file is a part of ChASE.
! Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
!   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
! License is 3-clause BSD:
! https://github.com/ChASE-library/ChASE

PROGRAM main
use chase_diag

integer ierr, init, i, j, k
integer N, nev, nex, idx_max
real(8) :: perturb, tmp, PI
real(8) :: tol
complex(8) :: cv
integer :: deg
character        :: mode, opt, qr
complex(8),  allocatable :: h(:,:), v(:,:)
real(8), allocatable :: lambda(:)

N = 1001
nev = 100
nex = 40
idx_max = 5
perturb = 1e-4

deg = 20
tol = 1e-10
mode = 'R'
opt = 'S'
qr = 'C'

print *, "ChASE Fortran example driver"

allocate(h(N, N)) 
allocate(v(N, nev+nex))
allocate(lambda(nev+nex))

call zchase_init(N, nev, nex, h, N, v, lambda, init)

! Generate Clement matrix
do i = 1, N
    h(i, i) = cmplx(0,0)
    do j = 1, N
        if (i .lt. N) then  ! Prevent accessing out-of-bounds
            tmp = real(i * (N + 1 - i))
            h(i+1, i) = cmplx(sqrt(tmp), 0)
        end if

        if (i .lt. N) then  ! Prevent accessing out-of-bounds
            tmp = real(i * (N + 1 - i))
            h(i, i + 1) = cmplx(sqrt(tmp), 0)
        end if
    end do
end do

do idx = 1, idx_max
	print *, "Starting Problem #", idx
	if(idx .ne. 0) then
		print *, "Using approximate solution"
	end if

	call zchase(deg, tol, mode, opt, qr)

	do i = 2, N
		do j = 2, i
		    call random_normal(tmp)
		    tmp = tmp * perturb
		    cv = cmplx(tmp, tmp)
		    h(j, i) = h(j,i) + cv
		    h(i, j) = h(i, j) + conjg(cv)
		end do 
	end do

	mode = 'A'
end do

call zchase_finalize(init)

contains

subroutine random_normal(randn)
	implicit none

	real(8),   intent(out) ::randn
	real(8) 			   ::rand, PI

	PI = 3.14159265358979323846

	call random_number(rand)
	randn = sqrt(-2.0 * log(rand)) * cos(2 * PI * rand)

end subroutine

END PROGRAM




