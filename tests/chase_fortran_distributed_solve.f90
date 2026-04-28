! This file is a part of ChASE.
! Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
!   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
! License is 3-clause BSD:
! https://github.com/ChASE-library/ChASE
!
! Fortran interface test analogous to chase_distributed_solve.cpp
! ChaseDistributedSolveGPUBlockBlockTest / CPU block-block:
!   N=1001, nev=100, nex=60; strict upper-triangle perturbation (deterministic hash
!   replacing C++ mt19937+normal); solver tol matches getResidualToleranceDist<T>().

program main
  use mpi
  use iso_fortran_env, only: real32, real64
  use chase_diag

  implicit none
  integer :: ierr, rank, size, comm
  integer :: nn, nev, nex
  integer :: dims(2), m, n, xoff, yoff, xlen, ylen
  integer :: init, deg
  real(real64) :: tmp
  character :: mode, opt, major, qr
  ! Match tests/chase_distributed_solve.cpp getResidualToleranceDist<T>()
  real(real32), parameter :: chase_tol_s = 1.0e-3_real32
  real(real64), parameter :: chase_tol_d = 1.0e-8_real64
  ! Match GPU (and same-size) block-block Clement perturb scale
  real(real32), parameter :: perturb_s = 1.0e-3_real32
  real(real64), parameter :: perturb_d = 1.0e-6_real64

  ! ChaseDistributedSolveGPUBlockBlockTest sizes (see chase_distributed_solve.cpp)
  nn = 1001
  nev = 100
  nex = 60
  deg = 16
  mode = 'R'
  opt = 'S'
  major = 'C'
  qr = 'C'

  call mpi_init(ierr)
  call mpi_comm_size(MPI_COMM_WORLD, size, ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
  comm = MPI_COMM_WORLD

  dims(1) = 0
  dims(2) = 0
  call mpi_dims_create(size, 2, dims, ierr)
  if (mod(nn, dims(1)) == 0) then
    m = nn / dims(1)
  else
    m = min(nn, nn / dims(1) + 1)
  end if
  if (mod(nn, dims(2)) == 0) then
    n = nn / dims(2)
  else
    n = min(nn, nn / dims(2) + 1)
  end if
  xoff = mod(rank, dims(1)) * m
  yoff = (rank / dims(1)) * n
  xlen = m
  ylen = n

  call test_pschase()
  call test_pdchase()
  call test_pcchase()
  call test_pzchase()

  call mpi_finalize(ierr)
  stop 0

contains

  subroutine test_pschase()
    real(real32), allocatable :: h(:,:), v(:,:)
    real(real32), allocatable :: ritzv(:)
    allocate(h(m, n), v(m, nev+nex), ritzv(nev+nex))
    h = 0.0_real32
    v = 0.0_real32
    ritzv = 0.0_real32
    call fill_clement_local_real_s(h, m, n, xoff, yoff, xlen, ylen, nn)
    call perturb_symm_offdiag_real_s(h, m, n, xoff, yoff, xlen, ylen, perturb_s)
    call pschase_init(nn, nev, nex, m, n, h, m, v, ritzv, dims(1), dims(2), major, comm, init)
    call pschase(deg, chase_tol_s, mode, opt, qr)
    if (rank == 0) call check_ritz_finite_s(ritzv, nev)
    call pschase_finalize(init)
    deallocate(h, v, ritzv)
  end subroutine test_pschase

  subroutine test_pdchase()
    real(real64), allocatable :: h(:,:), v(:,:)
    real(real64), allocatable :: ritzv(:)
    allocate(h(m, n), v(m, nev+nex), ritzv(nev+nex))
    h = 0.0_real64
    v = 0.0_real64
    ritzv = 0.0_real64
    call fill_clement_local_real_d(h, m, n, xoff, yoff, xlen, ylen, nn)
    call perturb_symm_offdiag_real_d(h, m, n, xoff, yoff, xlen, ylen, perturb_d)
    call pdchase_init(nn, nev, nex, m, n, h, m, v, ritzv, dims(1), dims(2), major, comm, init)
    call pdchase(deg, chase_tol_d, mode, opt, qr)
    if (rank == 0) call check_ritz_finite_d(ritzv, nev)
    call pdchase_finalize(init)
    deallocate(h, v, ritzv)
  end subroutine test_pdchase

  subroutine test_pcchase()
    complex(real32), allocatable :: h(:,:), v(:,:)
    real(real32), allocatable :: ritzv(:)
    allocate(h(m, n), v(m, nev+nex), ritzv(nev+nex))
    h = (0.0_real32, 0.0_real32)
    v = (0.0_real32, 0.0_real32)
    ritzv = 0.0_real32
    call fill_clement_local_cmplx_s(h, m, n, xoff, yoff, xlen, ylen, nn)
    call perturb_symm_offdiag_cmplx_s(h, m, n, xoff, yoff, xlen, ylen, perturb_s)
    call pcchase_init(nn, nev, nex, m, n, h, m, v, ritzv, dims(1), dims(2), major, comm, init)
    call pcchase(deg, chase_tol_s, mode, opt, qr)
    if (rank == 0) call check_ritz_finite_s(ritzv, nev)
    call pcchase_finalize(init)
    deallocate(h, v, ritzv)
  end subroutine test_pcchase

  subroutine test_pzchase()
    complex(real64), allocatable :: h(:,:), v(:,:)
    real(real64), allocatable :: ritzv(:)
    allocate(h(m, n), v(m, nev+nex), ritzv(nev+nex))
    h = (0.0_real64, 0.0_real64)
    v = (0.0_real64, 0.0_real64)
    ritzv = 0.0_real64
    call fill_clement_local_cmplx_d(h, m, n, xoff, yoff, xlen, ylen, nn)
    call perturb_symm_offdiag_cmplx_d(h, m, n, xoff, yoff, xlen, ylen, perturb_d)
    call pzchase_init(nn, nev, nex, m, n, h, m, v, ritzv, dims(1), dims(2), major, comm, init)
    call pzchase(deg, chase_tol_d, mode, opt, qr)
    if (rank == 0) call check_ritz_finite_d(ritzv, nev)
    call pzchase_finalize(init)
    deallocate(h, v, ritzv)
  end subroutine test_pzchase

  ! Deterministic U(0,1) from global indices (reproducible across ranks; not C++-identical).
  pure function det_uniform_01(ir, ic, k) result(u)
    integer, intent(in) :: ir, ic, k
    real(real64) :: u
    integer :: s
    s = mod(ir * 10009 + ic * 160081 + k * 104729, 2147483646) + 1
    u = real(s, real64) / 2147483647.0_real64
  end function det_uniform_01

  pure function normal01_det(ir, ic) result(z)
    integer, intent(in) :: ir, ic
    real(real64) :: z, u1, u2, twopi
    twopi = 8.0_real64 * atan(1.0_real64)
    u1 = max(det_uniform_01(ir, ic, 1), 1.0d-12)
    u2 = det_uniform_01(ir, ic, 2)
    z = sqrt(-2.0_real64 * log(u1)) * cos(twopi * u2)
  end function normal01_det

  subroutine perturb_symm_offdiag_real_s(h, m, n, xoff, yoff, xlen, ylen, perturb)
    integer, intent(in) :: m, n, xoff, yoff, xlen, ylen
    real(real32), intent(in) :: perturb
    real(real32), intent(inout) :: h(m, n)
    integer :: x, y, x_g, y_g, ir, ic
    real(real64) :: ep
    do x = 1, xlen
      do y = 1, ylen
        x_g = xoff + x
        y_g = yoff + y
        if (x_g == y_g) cycle
        ir = min(x_g, y_g)
        ic = max(x_g, y_g)
        ep = normal01_det(ir, ic)
        h(x, y) = h(x, y) + perturb * real(ep, real32)
      end do
    end do
  end subroutine perturb_symm_offdiag_real_s

  subroutine perturb_symm_offdiag_real_d(h, m, n, xoff, yoff, xlen, ylen, perturb)
    integer, intent(in) :: m, n, xoff, yoff, xlen, ylen
    real(real64), intent(in) :: perturb
    real(real64), intent(inout) :: h(m, n)
    integer :: x, y, x_g, y_g, ir, ic
    real(real64) :: ep
    do x = 1, xlen
      do y = 1, ylen
        x_g = xoff + x
        y_g = yoff + y
        if (x_g == y_g) cycle
        ir = min(x_g, y_g)
        ic = max(x_g, y_g)
        ep = normal01_det(ir, ic)
        h(x, y) = h(x, y) + perturb * ep
      end do
    end do
  end subroutine perturb_symm_offdiag_real_d

  subroutine perturb_symm_offdiag_cmplx_s(h, m, n, xoff, yoff, xlen, ylen, perturb)
    integer, intent(in) :: m, n, xoff, yoff, xlen, ylen
    real(real32), intent(in) :: perturb
    complex(real32), intent(inout) :: h(m, n)
    integer :: x, y, x_g, y_g, ir, ic
    complex(real32) :: ep
    real(real64) :: nr, ni
    do x = 1, xlen
      do y = 1, ylen
        x_g = xoff + x
        y_g = yoff + y
        if (x_g == y_g) cycle
        ir = min(x_g, y_g)
        ic = max(x_g, y_g)
        nr = normal01_det(ir, ic)
        ni = normal01_det(ic, ir)
        ep = cmplx(perturb * real(nr, real32), perturb * real(ni, real32), real32)
        if (x_g < y_g) then
          h(x, y) = h(x, y) + ep
        else
          h(x, y) = h(x, y) + conjg(ep)
        end if
      end do
    end do
  end subroutine perturb_symm_offdiag_cmplx_s

  subroutine perturb_symm_offdiag_cmplx_d(h, m, n, xoff, yoff, xlen, ylen, perturb)
    integer, intent(in) :: m, n, xoff, yoff, xlen, ylen
    real(real64), intent(in) :: perturb
    complex(real64), intent(inout) :: h(m, n)
    integer :: x, y, x_g, y_g, ir, ic
    complex(real64) :: ep
    real(real64) :: nr, ni
    do x = 1, xlen
      do y = 1, ylen
        x_g = xoff + x
        y_g = yoff + y
        if (x_g == y_g) cycle
        ir = min(x_g, y_g)
        ic = max(x_g, y_g)
        nr = normal01_det(ir, ic)
        ni = normal01_det(ic, ir)
        ep = cmplx(perturb * nr, perturb * ni, real64)
        if (x_g < y_g) then
          h(x, y) = h(x, y) + ep
        else
          h(x, y) = h(x, y) + conjg(ep)
        end if
      end do
    end do
  end subroutine perturb_symm_offdiag_cmplx_d

  subroutine fill_clement_local_real_s(h, m, n, xoff, yoff, xlen, ylen, nn)
    integer, intent(in) :: m, n, xoff, yoff, xlen, ylen, nn
    real(real32), intent(inout) :: h(m, n)
    integer :: x, y, x_g, y_g
    real(real64) :: tmp
    do x = 1, xlen
      do y = 1, ylen
        x_g = xoff + x
        y_g = yoff + y
        if (x_g == y_g + 1) then
          tmp = real(y_g * (nn + 1 - y_g), real64)
          h(x, y) = real(sqrt(tmp), real32)
        end if
        if (y_g == x_g + 1) then
          tmp = real(x_g * (nn + 1 - x_g), real64)
          h(x, y) = real(sqrt(tmp), real32)
        end if
      end do
    end do
  end subroutine fill_clement_local_real_s

  subroutine fill_clement_local_real_d(h, m, n, xoff, yoff, xlen, ylen, nn)
    integer, intent(in) :: m, n, xoff, yoff, xlen, ylen, nn
    real(real64), intent(inout) :: h(m, n)
    integer :: x, y, x_g, y_g
    real(real64) :: tmp
    do x = 1, xlen
      do y = 1, ylen
        x_g = xoff + x
        y_g = yoff + y
        if (x_g == y_g + 1) then
          tmp = real(y_g * (nn + 1 - y_g), real64)
          h(x, y) = sqrt(tmp)
        end if
        if (y_g == x_g + 1) then
          tmp = real(x_g * (nn + 1 - x_g), real64)
          h(x, y) = sqrt(tmp)
        end if
      end do
    end do
  end subroutine fill_clement_local_real_d

  subroutine fill_clement_local_cmplx_s(h, m, n, xoff, yoff, xlen, ylen, nn)
    integer, intent(in) :: m, n, xoff, yoff, xlen, ylen, nn
    complex(real32), intent(inout) :: h(m, n)
    integer :: x, y, x_g, y_g
    real(real64) :: tmp
    do x = 1, xlen
      do y = 1, ylen
        x_g = xoff + x
        y_g = yoff + y
        if (x_g == y_g + 1) then
          tmp = real(y_g * (nn + 1 - y_g), real64)
          h(x, y) = cmplx(sqrt(tmp), 0.0_real64, real32)
        end if
        if (y_g == x_g + 1) then
          tmp = real(x_g * (nn + 1 - x_g), real64)
          h(x, y) = cmplx(sqrt(tmp), 0.0_real64, real32)
        end if
      end do
    end do
  end subroutine fill_clement_local_cmplx_s

  subroutine fill_clement_local_cmplx_d(h, m, n, xoff, yoff, xlen, ylen, nn)
    integer, intent(in) :: m, n, xoff, yoff, xlen, ylen, nn
    complex(real64), intent(inout) :: h(m, n)
    integer :: x, y, x_g, y_g
    real(real64) :: tmp
    do x = 1, xlen
      do y = 1, ylen
        x_g = xoff + x
        y_g = yoff + y
        if (x_g == y_g + 1) then
          tmp = real(y_g * (nn + 1 - y_g), real64)
          h(x, y) = cmplx(sqrt(tmp), 0.0_real64, real64)
        end if
        if (y_g == x_g + 1) then
          tmp = real(x_g * (nn + 1 - x_g), real64)
          h(x, y) = cmplx(sqrt(tmp), 0.0_real64, real64)
        end if
      end do
    end do
  end subroutine fill_clement_local_cmplx_d

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
