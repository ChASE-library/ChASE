! This file is a part of ChASE.
! Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
!   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
! License is 3-clause BSD:
! https://github.com/ChASE-library/ChASE
!
! Fortran interface test aligned with the C++ serial integration tests:
! single-rank pure Clement matrix, s/d/c/z chase init/solve/finalize.
! Sequence of solves with approximate mode after the first (no matrix perturbation).
! After the final solve: compute residuals ||H*v_i - lambda_i*v_i|| from stored H, v, lambda;
! require residuals < tol and residuals /= 0 (CPU/GPU agnostic).

program main
  use mpi
  use iso_fortran_env, only: real32, real64
  use chase_diag

  implicit none
  integer :: ierr, rank
  integer :: n, nev, nex, ldh, idx_max, idx
  integer :: init
  character :: mode, opt, qr
  ! Solver settings: single precision uses looser tolerance.
  integer, parameter :: deg_s = 16
  integer, parameter :: deg_d = 20
  real(real32), parameter :: chase_tol_s = 1.0e-4_real32
  real(real64), parameter :: chase_tol_d = 1.0e-8_real64
  ! Residual check tolerances (match getResidualTolerance: single 1e-3, double 1e-8)
  real(real32), parameter :: resid_tol_s = 1.0e-3_real32
  real(real64), parameter :: resid_tol_d = 1.0e-8_real64

  ! Keep this as a stable Fortran interface smoke test.  The heavier Clement
  ! cases are covered by the C++ integration tests and examples.
  n = 256
  ldh = n
  nev = 24
  nex = 16
  idx_max = 2
  mode = 'R'
  opt = 'S'
  qr = 'C'

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)

  if (rank /= 0) then
    call mpi_finalize(ierr)
    stop 0
  end if

  ! Run tests for all four precisions (s, d, c, z)
  call test_schase()
  call test_dchase()
  call test_cchase()
  call test_zchase()

  call mpi_finalize(ierr)
  stop 0

contains

  subroutine test_schase()
    real(real32), allocatable :: h(:,:), v(:,:), resid(:)
    real(real32), allocatable :: ritzv(:)
    allocate(h(ldh, n), v(n, nev+nex), ritzv(nev+nex), resid(nev))
    call build_clement_real_s(h, ldh, n)
    v = 0.0_real32
    ritzv = 0.0_real32
    call schase_init(n, nev, nex, h, ldh, v, ritzv, init)
    mode = 'R'
    do idx = 1, idx_max
      call schase(deg_s, chase_tol_s, mode, opt, qr)
      if (idx < idx_max) mode = 'A'
    end do
    call check_ritz_finite_s(ritzv, nev)
    call compute_residuals_s(n, ldh, h, nev, v, n, ritzv, resid)
    call check_residuals_s(nev, resid, resid_tol_s)
    call schase_finalize(init)
    deallocate(h, v, ritzv, resid)
  end subroutine test_schase

  subroutine test_dchase()
    real(real64), allocatable :: h(:,:), v(:,:), resid(:)
    real(real64), allocatable :: ritzv(:)
    allocate(h(ldh, n), v(n, nev+nex), ritzv(nev+nex), resid(nev))
    call build_clement_real_d(h, ldh, n)
    v = 0.0_real64
    ritzv = 0.0_real64
    call dchase_init(n, nev, nex, h, ldh, v, ritzv, init)
    mode = 'R'
    do idx = 1, idx_max
      call dchase(deg_d, chase_tol_d, mode, opt, qr)
      if (idx < idx_max) mode = 'A'
    end do
    call check_ritz_finite_d(ritzv, nev)
    call compute_residuals_d(n, ldh, h, nev, v, n, ritzv, resid)
    call check_residuals_d(nev, resid, resid_tol_d)
    call dchase_finalize(init)
    deallocate(h, v, ritzv, resid)
  end subroutine test_dchase

  subroutine test_cchase()
    complex(real32), allocatable :: h(:,:), v(:,:)
    real(real32), allocatable :: ritzv(:), resid(:)
    allocate(h(ldh, n), v(n, nev+nex), ritzv(nev+nex), resid(nev))
    call build_clement_cmplx_s(h, ldh, n)
    v = (0.0_real32, 0.0_real32)
    ritzv = 0.0_real32
    call cchase_init(n, nev, nex, h, ldh, v, ritzv, init)
    mode = 'R'
    do idx = 1, idx_max
      call cchase(deg_s, chase_tol_s, mode, opt, qr)
      if (idx < idx_max) mode = 'A'
    end do
    call check_ritz_finite_s(ritzv, nev)
    call compute_residuals_c(n, ldh, h, nev, v, n, ritzv, resid)
    call check_residuals_s(nev, resid, resid_tol_s)
    call cchase_finalize(init)
    deallocate(h, v, ritzv, resid)
  end subroutine test_cchase

  subroutine test_zchase()
    complex(real64), allocatable :: h(:,:), v(:,:)
    real(real64), allocatable :: ritzv(:), resid(:)
    allocate(h(ldh, n), v(n, nev+nex), ritzv(nev+nex), resid(nev))
    call build_clement_cmplx_d(h, ldh, n)
    v = (0.0_real64, 0.0_real64)
    ritzv = 0.0_real64
    call zchase_init(n, nev, nex, h, ldh, v, ritzv, init)
    mode = 'R'
    do idx = 1, idx_max
      call zchase(deg_d, chase_tol_d, mode, opt, qr)
      if (idx < idx_max) mode = 'A'
    end do
    call check_ritz_finite_d(ritzv, nev)
    call compute_residuals_z(n, ldh, h, nev, v, n, ritzv, resid)
    call check_residuals_d(nev, resid, resid_tol_d)
    call zchase_finalize(init)
    deallocate(h, v, ritzv, resid)
  end subroutine test_zchase

  subroutine build_clement_real_s(h, ldh, n)
    integer, intent(in) :: ldh, n
    real(real32), intent(inout) :: h(ldh, n)
    integer :: i
    h = 0.0_real32
    do i = 1, n
      h(i, i) = 0.0_real32
      if (i < n) then
        h(i+1, i) = real(sqrt(real(i*(n+1-i), real64)), real32)
        h(i, i+1) = h(i+1, i)
      end if
    end do
  end subroutine build_clement_real_s

  subroutine build_clement_real_d(h, ldh, n)
    integer, intent(in) :: ldh, n
    real(real64), intent(inout) :: h(ldh, n)
    integer :: i
    h = 0.0_real64
    do i = 1, n
      h(i, i) = 0.0_real64
      if (i < n) then
        h(i+1, i) = sqrt(real(i*(n+1-i), real64))
        h(i, i+1) = h(i+1, i)
      end if
    end do
  end subroutine build_clement_real_d

  subroutine build_clement_cmplx_s(h, ldh, n)
    integer, intent(in) :: ldh, n
    complex(real32), intent(inout) :: h(ldh, n)
    integer :: i
    real(real64) :: tmp
    h = (0.0_real32, 0.0_real32)
    do i = 1, n
      h(i, i) = (0.0_real32, 0.0_real32)
      if (i < n) then
        tmp = real(i*(n+1-i), real64)
        h(i+1, i) = cmplx(sqrt(tmp), 0.0_real64, real32)
        h(i, i+1) = h(i+1, i)
      end if
    end do
  end subroutine build_clement_cmplx_s

  subroutine build_clement_cmplx_d(h, ldh, n)
    integer, intent(in) :: ldh, n
    complex(real64), intent(inout) :: h(ldh, n)
    integer :: i
    real(real64) :: tmp
    h = (0.0_real64, 0.0_real64)
    do i = 1, n
      h(i, i) = (0.0_real64, 0.0_real64)
      if (i < n) then
        tmp = real(i*(n+1-i), real64)
        h(i+1, i) = cmplx(sqrt(tmp), 0.0_real64, real64)
        h(i, i+1) = h(i+1, i)
      end if
    end do
  end subroutine build_clement_cmplx_d

  ! Compute residuals resid(i) = || H*v_i - lambda_i*v_i ||_2 (same as C++ residuals())
  subroutine compute_residuals_s(n, ldh, h, nev, v, ldv, ritzv, resid)
    integer, intent(in) :: n, ldh, nev, ldv
    real(real32), intent(in) :: h(ldh, n), v(ldv, nev), ritzv(nev)
    real(real32), intent(out) :: resid(nev)
    real(real32), allocatable :: w(:)
    integer :: i, k, j
    allocate(w(n))
    do i = 1, nev
      w = 0.0_real32
      do j = 1, n
        do k = 1, n
          w(k) = w(k) + h(k, j) * v(j, i)
        end do
      end do
      do k = 1, n
        w(k) = w(k) - ritzv(i) * v(k, i)
      end do
      resid(i) = 0.0_real32
      do k = 1, n
        resid(i) = resid(i) + w(k) * w(k)
      end do
      resid(i) = sqrt(resid(i))
    end do
    deallocate(w)
  end subroutine compute_residuals_s

  subroutine compute_residuals_d(n, ldh, h, nev, v, ldv, ritzv, resid)
    integer, intent(in) :: n, ldh, nev, ldv
    real(real64), intent(in) :: h(ldh, n), v(ldv, nev), ritzv(nev)
    real(real64), intent(out) :: resid(nev)
    real(real64), allocatable :: w(:)
    integer :: i, k, j
    allocate(w(n))
    do i = 1, nev
      w = 0.0_real64
      do j = 1, n
        do k = 1, n
          w(k) = w(k) + h(k, j) * v(j, i)
        end do
      end do
      do k = 1, n
        w(k) = w(k) - ritzv(i) * v(k, i)
      end do
      resid(i) = 0.0_real64
      do k = 1, n
        resid(i) = resid(i) + w(k) * w(k)
      end do
      resid(i) = sqrt(resid(i))
    end do
    deallocate(w)
  end subroutine compute_residuals_d

  subroutine compute_residuals_c(n, ldh, h, nev, v, ldv, ritzv, resid)
    integer, intent(in) :: n, ldh, nev, ldv
    complex(real32), intent(in) :: h(ldh, n), v(ldv, nev)
    real(real32), intent(in) :: ritzv(nev)
    real(real32), intent(out) :: resid(nev)
    complex(real32), allocatable :: w(:)
    integer :: i, k, j
    allocate(w(n))
    do i = 1, nev
      w = (0.0_real32, 0.0_real32)
      do j = 1, n
        do k = 1, n
          w(k) = w(k) + h(k, j) * v(j, i)
        end do
      end do
      do k = 1, n
        w(k) = w(k) - ritzv(i) * v(k, i)
      end do
      resid(i) = 0.0_real32
      do k = 1, n
        resid(i) = resid(i) + real(conjg(w(k))*w(k), real32)
      end do
      resid(i) = sqrt(resid(i))
    end do
    deallocate(w)
  end subroutine compute_residuals_c

  subroutine compute_residuals_z(n, ldh, h, nev, v, ldv, ritzv, resid)
    integer, intent(in) :: n, ldh, nev, ldv
    complex(real64), intent(in) :: h(ldh, n), v(ldv, nev)
    real(real64), intent(in) :: ritzv(nev)
    real(real64), intent(out) :: resid(nev)
    complex(real64), allocatable :: w(:)
    integer :: i, k, j
    allocate(w(n))
    do i = 1, nev
      w = (0.0_real64, 0.0_real64)
      do j = 1, n
        do k = 1, n
          w(k) = w(k) + h(k, j) * v(j, i)
        end do
      end do
      do k = 1, n
        w(k) = w(k) - ritzv(i) * v(k, i)
      end do
      resid(i) = 0.0_real64
      do k = 1, n
        resid(i) = resid(i) + real(conjg(w(k))*w(k), real64)
      end do
      resid(i) = sqrt(resid(i))
    end do
    deallocate(w)
  end subroutine compute_residuals_z

  subroutine check_residuals_s(nev, resid, resid_tol)
    integer, intent(in) :: nev
    real(real32), intent(in) :: resid(nev), resid_tol
    integer :: i
    do i = 1, nev
      if (resid(i) >= resid_tol) stop 1
      if (resid(i) == 0.0_real32) stop 1
    end do
  end subroutine check_residuals_s

  subroutine check_residuals_d(nev, resid, resid_tol)
    integer, intent(in) :: nev
    real(real64), intent(in) :: resid(nev), resid_tol
    integer :: i
    do i = 1, nev
      if (resid(i) >= resid_tol) stop 1
      if (resid(i) == 0.0_real64) stop 1
    end do
  end subroutine check_residuals_d

  subroutine check_ritz_finite_s(ritzv, nev)
    integer, intent(in) :: nev
    real(real32), intent(in) :: ritzv(*)
    integer :: i
    do i = 1, min(5, nev)
      if (.not. (abs(ritzv(i)) >= 0.0_real32 .and. abs(ritzv(i)) <= huge(1.0_real32))) then
        stop 1
      end if
      if (ritzv(i) /= ritzv(i)) then
        stop 1
      end if
    end do
  end subroutine check_ritz_finite_s

  subroutine check_ritz_finite_d(ritzv, nev)
    integer, intent(in) :: nev
    real(real64), intent(in) :: ritzv(*)
    integer :: i
    do i = 1, min(5, nev)
      if (.not. (abs(ritzv(i)) >= 0.0_real64 .and. abs(ritzv(i)) <= huge(1.0_real64))) then
        stop 1
      end if
      if (ritzv(i) /= ritzv(i)) then
        stop 1
      end if
    end do
  end subroutine check_ritz_finite_d

end program main
