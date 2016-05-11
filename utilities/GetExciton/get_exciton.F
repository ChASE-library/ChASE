!************************************************************************
! This program is part of the BSE code developed at the IFTO 
! (University of Jena).
!
! Authors: Patrick Hahn
!          Andreas Hermann
!          Claudia Roedl
!          Frank Fuchs
!
! It requires as input quasiparticle energies, optical transition 
! matrix elements, and the matrix elements of the BSE Hamiltonian.
! Usually they are calculated by means of some additional subroutines in
! vasp.4.4 which were originally developed by Juergen Furthmueller and 
! Patrick Hahn and later modified by the authors listed above.
!
! This program offers several possibilities:
!  - bse_mode=DA: full diagonalization of the Hamiltonian yielding
!    spectra, eigenvalues, and eigenfunctions
!  - bse_mode=CG: iterative diagonalization of the Hamiltonian yielding
!    the lowest eigenvalues and eigenfunctions
!  - bse_mode=TD: transformation of the eigenvalue problem to an initial-
!    value problem yielding spectra
!  - calculation of RPA spectra is supported
!  - collinear spin polarization is supported
!  - some routines for the visualization of the excitonic wave function
!    are provided
!
! Publications offering a detailed description of the implementation and 
! the algorithms:
!   Hahn et al., PRL 88, 016402 (2002)
!   Schmidt et al., PRB 67, 085307 (2003)
!   Roedl et al., PRB 77, 184408 (2008)
!   Fuchs et al., PRB 78, 085103 (2008) 
!
!************************************************************************

!************************************************************************

!************************************************************************
!     These modules are thought of as a library for routines + algorithms 
!     for the calculation of BSE-spectra.
!     Please do not combine the modules excivar and exci_helper since 
!     seperating them yields additional error checks.
!************************************************************************

!************************************************************************
!     Here you are given the opportunity to select the precision for the 
!     following calculations. With only a few exceptions all quantities 
!     should be defined as real/complex(mq).
!     Please do not use the helper.F q !!
!************************************************************************

module excivar
      implicit none

!#define prec_double
#ifdef prec_double
      INTEGER, PARAMETER :: mq=SELECTED_REAL_KIND(10)
      INTEGER, PARAMETER :: lq=SELECTED_REAL_KIND(10)
#else
      INTEGER, PARAMETER :: mq=SELECTED_REAL_KIND(4)
      INTEGER, PARAMETER :: lq=SELECTED_REAL_KIND(4)
#endif
!------------------------------------------------------------------------
#define MAT_S    1
#define MAT_M    2
!------------------------------------------------------------------------
#define infwrite        write(*,*)
#define infwritef       write
#define iinfwrite       !! write(*,*)
#define iinfwritef      !! write
#define do_io 
#define idwrite         !
#define mdwrite         write(*,*)
#define dwrite          !write(*,*)
! One can exploit the fact that the BSE is hermitian to gain some speed 
! of roughly a factor of 2.
#define use_herm
!------------------------------------------------------------------------
! Variables for the BSE stuff - no counters please!!
      integer, parameter :: nbfield=256
! Some parameters read from input file.
! Control flags.
      integer  :: exciton              ! (1) BSE, (0) RPA
      integer  :: lfcorr               ! (1) with, (0) without LFEs (only important if ispin=2) (should remain 1 now in all cases)
      integer  :: beyondtd             ! (1) going beyond, (0) sticking with Tamm-Dancoff approximation
      integer  :: checkposdef          ! (1) beyond Tamm-Dancoff Hamiltonian FH transformed to H to check positive definiteness, (0) off
      integer  :: spinflip             ! (1) spin-flip excitations cu->vd, (2) spin-flip excitations cd->vu, (0) ordinary spectrum
      integer  :: ispin                ! Number of spin components (for spin flips this is one) ... just for Claudia :))
      integer, pointer :: nbval(:)     ! nbval(ispin) -> number of valence bands.
      integer, pointer :: nbcon(:)     ! nbcon(ispin) number of conduction bands.
      integer  :: maxk                 ! Number of k-points.
      integer  :: nthreads_ma          ! Number of files the matrix is distributed over.
      integer  :: nexciband_ma         ! Number of relevant entries in exciband_ma.
      integer  :: nexciband_sp         ! Number of relevant entries in exciband_sp.
      integer  :: exciband_ma(nbfield) ! Contains band index list of bands contributing to the matrix.
      integer  :: exciband_sp(nbfield) ! Contains band index list of bands contributing to the spectra.
      real(mq) :: encut_ma             ! Energy cutoff which was used for the calculation of the matrix.
      real(mq) :: encut_sp             ! Energy cutoff to be used for the calculation of the spectra.
      real(mq) :: qpshiftsci           ! Scissors shift for QP corrections.
      character (255) :: gw_file       ! File name for GW corrections.
! Spectra.
      integer  :: xyzinfo              ! Directions for spectra: (1) x, (2) y, (3) z, (0) all
      integer  :: ngamma               ! Number of different broadenings.
      real(mq) :: cellvol              ! Unit cell volume in Bohr^3.
      real(mq) :: gamma(nbfield)       ! Broadening for the spectra in eV.
      character(255) :: spectra_file   ! File name for calculated spectra.
! TD.
      integer  :: ntdsteps             ! Maximum number of time steps for the TD.
      integer  :: last_tdstep          ! Number of the last actual time step in the TD.
      real(mq) :: evmax                ! Upper bound for eigenvalues in TD. Usually 1.5*encut_sp.
! CG and DA.
      integer  :: nevcg                ! Number of calculated eigenvalues.
      integer  :: nevcgadd             ! Some additional eigenvalues for better convergence.
      integer  :: nevneg               ! Number of negative eigenvalues in beyond Tamm-Dancoff mode.
      integer  :: nevpairs             ! Number of +/- E_Lambda eigenvalue pairs in beyond Tamm-Dancoff mode.
      integer  :: nksmax               ! Maximum number of KS iterations.
      integer  :: ncgmin               ! Minimum number of CG iterations.
      integer  :: ncgmax               ! Maximum number of CG iterations.
      integer  :: ncgrenorm            ! Number of CG steps between frequency renormalizations.
      real(mq) :: rsd_r                ! Target relative residual error.
      real(mq) :: rsd_a                ! Target absolute residual error.
      real(mq) :: fsenergy             ! Folded spectrum reference energy.
      real(mq) :: fsshift              ! Folded spectrum (small) energy shift to get rid of degeneracies introduced by squaring the matrix.
      real(mq) :: fsoffset             ! Folded spectrum offset energy.
      character(255) :: rpa_ev_file    ! File name for calculated RPA eigenvalues and oscillator strengths.
      character(255) :: exci_ev_file   ! File name for calculated excitonic eigenvalues and oscillator strengths.
      character(255) :: exci_ef_file   ! File name for calculated excitonic eigenfunctions.
! Will be ruled out sooner or later.
      integer  :: koffset              ! Do not include the first koffset k-points. (No longer used.)
! Some other variables.
      integer  :: nbands               ! nbval+nbcon
      integer  :: nbvalmax             ! Maximum number of valence bands in the spin channels. 
      integer  :: nbconmax             ! Maximum number of conduction bands in the spin channels.
      integer  :: ispinall             ! ispinall==ispin except for spinflips where ispinall==2*ispin
      integer  :: matdim               ! Matrix dimension used for the calculation.
      integer  :: matdim_ma            ! Matrix dimension on disk.
      integer  :: matdim_node(0:255)   ! Number of rows at each node.
      integer  :: matstart(0:255)      ! Matrix element after which matrix at each node begins.
      integer  :: matdim_btd           ! Matrix dimension used for the calculation in beyond Tamm-Dancoff mode.
!      real(mq) :: ecs(100)             ! Multiple cutoffs within one run. (Currently not supported.)
!      integer  :: nec=1
! Some arrays.
      integer, pointer     :: nehmen_ma(:,:,:,:)     ! nehmen_ma(nbvalmax,nbconmax,maxk,ispin) -> transitions included in matrix on disk
      integer, pointer     :: nehmen(:,:,:,:)        ! nehmen(nbvalmax,nbconmax,maxk,ispin) -> transitions used for calculation / matrix in storage
      integer, pointer     :: nehmen_loc(:,:,:,:)    ! nehmen_loc(0:nbvalmax,0:nbconmax,maxk,ispin) -> transitions stored on local node
      integer, pointer     :: nehmen_loc_ar(:,:,:,:) ! nehmen_loc_ar(0:nbvalmax,0:nbconmax,maxk,ispin) -> antiresonant transitions stored on local node
      integer, pointer     :: nehmen_b_ma(:)         ! nehmen_b_ma(nbands) -> bands included in matrix on disk
      integer, pointer     :: nehmen_b_sp(:)         ! nehmen_b_sp(nbands) -> bands included in calculation
      integer, pointer     :: dim_at_k_ma(:,:)       ! dim_at_k_ma(maxk,ispin) -> dimension for each k-point and spin on disk
      integer, pointer     :: dim_at_k(:,:)          ! dim_at_k(maxk,ispin) -> dimension for each k-point and spin
      integer, pointer     :: cvk_tr_ma(:,:,:,:)     ! cvk_tr_ma(nbvalmax,nbconmax,maxk,ispin) -> matrix index of a transition for matrix on disk
      integer, pointer     :: cvk_tr_sp(:,:,:,:)     ! cvk_tr_sp(nbvalmax,nbconmax,maxk,ispin) -> matrix index of a transition in the calculation
      integer, pointer     :: cvk_artr_sp(:,:,:,:)   ! cvk_artr_sp(nbvalmax,nbconmax,maxk,ispin) -> matrix index of an antiresonant transition in the calculation
      real(mq), pointer    :: gw(:,:,:)              ! gw(nbands,maxk,ispinall) (ispinall==2 also for spinflips) -> GW shifts as read from file
      real(mq), pointer    :: ldagw(:)               ! ldagw(matdim) -> QP eigenvalue differences for main diagonal
      real(mq), pointer    :: kptweight(:)           ! kptweight(matdim) -> respective k-point weight for each transition
      real(mq), pointer    :: bse_eval(:)            ! bse_eval(matdim) -> list of eigenvalues
      real(mq), pointer    :: timestep_v(:,:)        ! timestep_v(2,ntdsteps) -> time steps and cumulated time steps for TD
      complex(mq), pointer :: bsematrix(:,:)         ! bsematrix(matdim,matdim) -> Guess, ... ... the big elephant, of course.
      complex(mq), pointer :: optmat(:,:)            ! optmat(matdim,3) -> optical transition matrix element for each transition in x, y, and z direction
      complex(mq), pointer :: scalar(:)              ! scalar(ntdsteps) -> scalar product for each time step (main quantity in TD) 
      complex(mq), pointer :: bse_evec(:,:)          ! bse_evec(matdim,nevcg+nevcgadd) -> excitonic eigenfunctions
! Some standard file names.
      character(255) :: input_file='input_exciton_calc'
      character(255) :: optics_file(2)=(/'Transmatrix   ','Transmatrix_s2'/)
      character(255) :: matrix_file='MATRIXDATEI'
      character(255) :: matrix_btd_file='MATRIXDATEI_BTD'
      character(255) :: matrix_sfl_file(2)=(/'MATRIXDATEI_CUVD','MATRIXDATEI_CDVU'/)
      character(255) :: matrix_sfl_btd_file(2)=(/'MATRIXDATEI_CUVD_BTD','MATRIXDATEI_CDVU_BTD'/)
! Some constants.
      real(mq), parameter     :: pi       = 3.141592654_mq
      real(mq), parameter     :: autoa    = 0.529177249_mq
      real(mq), parameter     :: rytoev   = 13.605826_mq
      real(mq), parameter     :: hartree  = 2._mq*rytoev !27.2112_mq
      complex(mq), parameter  :: imun     = (0.0_mq,1.0_mq)
      complex(mq), parameter  :: inv_imun = (0.0_mq,-1.0_mq)
      complex(mq), parameter  :: cone     = (1.0_mq,0.0_mq)
      complex(mq), parameter  :: czero    = (0.0_mq,0.0_mq)
      integer :: node_lead ! Number of the leading process.
      integer :: node_me   ! Number of the my process.
      integer :: nodes     ! Total number of processes.

end module excivar

!************************************************************************

!************************************************************************
!     This module contains all subroutines.
!************************************************************************

      module excihelper
      implicit none
      
      contains

!************************************************************************
!     Routines for reading and building.
!************************************************************************

      subroutine read_parameters()
      use excivar
      implicit none
!local
      integer      :: idum,iu,n,ierr
      real(mq)      :: rdum
      real(mq)      :: rdump(nbfield)
      complex(mq)   :: cdum
      logical      :: ldum,lopen
      character(1) :: str

!     Control flags.
      exciton=1

      lfcorr=1

      beyondtd=0

      checkposdef=0

      spinflip=0

!     -------------------------------------------------------------------
!     Specifications for the matrix.
      ispin=1
      if (ispin.ne.2) ispin=1
      ispinall=ispin
      if (ispin.eq.2.and.spinflip.ne.0) ispin=1

      allocate(nbval(ispinall))
      nbval=44
      if (ispinall==2.and.nbval(ispinall)==0) nbval(ispinall)=nbval(1)

      allocate(nbcon(ispinall))
      nbcon=4
      if (ispinall==2.and.nbcon(ispinall)==0) nbcon(ispinall)=nbcon(1)

      maxk=7795

      nthreads_ma=640

      exciband_ma=0
      nexciband_ma=1
      do while (exciband_ma(nexciband_ma+1)>0.and.nexciband_ma<nbfield)
         nexciband_ma=nexciband_ma+1
      enddo

      exciband_sp=0
      nexciband_sp=1
      do while (exciband_sp(nexciband_sp+1)>0.and.nexciband_sp<nbfield)
         nexciband_sp=nexciband_sp+1
      enddo

      encut_ma=6._mq

      encut_sp=6._mq
      if (encut_sp>encut_ma) encut_sp=encut_ma

      qpshiftsci=2.3701_mq

!     -------------------------------------------------------------------
!     Spectra.

!     -------------------------------------------------------------------
!     Files
      gw_file='QPSHIFT'
      gw_file=trim(adjustl(gw_file))

      spectra_file='spectra'
      spectra_file=trim(adjustl(spectra_file))

      rpa_ev_file='rpa_eigenvals'
      rpa_ev_file=trim(adjustl(rpa_ev_file))

      exci_ev_file='exci_eigenvals'
      exci_ev_file=trim(adjustl(exci_ev_file))

      exci_ef_file='exci_eigenfunctions'
      exci_ef_file=trim(adjustl(exci_ef_file))

!     -------------------------------------------------------------------
!     Will be ruled out sooner or later.
      koffset=0

      end subroutine read_parameters


!     *******************************************************************
!     Find the dimension for the matrix the spectrum is calculated for.

      subroutine get_matdim_from_optics()
      use excivar
      implicit none
 ! local
      integer  :: count,spin,fspin,kpt,c,v,n,dum_c,dum_v
      real(mq) :: kweight,eig_c,eig_v,px_re,px_im,py_re,py_im,pz_re,pz_im,dum
      real(mq), pointer :: all_eig_c(:,:,:),all_eig_v(:,:,:)

!     Take into account only some specific bands if specified using
!     EXCIBANDMA and EXCIBANDSP.
      nehmen_b_ma=0
      if (exciband_ma(1).ne.0) then
         do n=1,nbfield
            if (exciband_ma(n).ne.0) nehmen_b_ma(exciband_ma(n))=1
         enddo
      elseif (exciband_ma(1).eq.0) then
         nehmen_b_ma=1
      endif
      nehmen_b_sp=0
      if (exciband_sp(1).ne.0) then
         do n=1,nbfield
            if (exciband_sp(n).ne.0) nehmen_b_sp(exciband_sp(n))=1
         enddo
      elseif (exciband_sp(1).eq.0) then
         nehmen_b_sp=1
      endif
      do n=1,nbands
         if (nehmen_b_ma(n)==0.and.nehmen_b_sp(n)==1) then
            infwritef(*,'(1x,(a),i3,(a))') 'Band ',n,' requested for spectra not present in matrix.' 
            stop
         endif
      enddo

!     Take into account only transitions below cutoff for spectra.
      if (spinflip==0) then
         count=0
         do spin=1,ispin
            open(20,file=optics_file(spin),action='read',status='old')
            do kpt=1,koffset
               do c=1,nbcon(spin)
                  do v=1,nbval(spin)
                     read(20,*)
                  enddo
               enddo
            enddo
            do kpt=1,maxk
               do c=1,nbcon(spin)
                  do v=1,nbval(spin)
                     read(20,*) kweight,dum_c,dum_v,eig_c,eig_v,px_re,px_im,py_re,py_im,pz_re,pz_im
                     if ((eig_c-eig_v)<encut_sp) then
                     if ((nehmen_b_sp(v)==1).and.(nehmen_b_sp(nbval(spin)+c)==1)) then
                        count=count+1
                     endif
                     endif
                  enddo
               enddo
            enddo
            close(20)
         enddo
         matdim=count
      elseif (spinflip==1.or.spinflip==2) then
         fspin=3-spinflip
         allocate(all_eig_c(nbconmax,maxk,ispinall))
         allocate(all_eig_v(nbvalmax,maxk,ispinall))
         all_eig_c=0._mq
         all_eig_v=0._mq
         open(20,file=optics_file(spinflip),action='read',status='old')
         open(21,file=optics_file(fspin),action='read',status='old')
         do kpt=1,koffset
            do c=1,nbcon(spinflip)
               do v=1,nbval(spinflip)
                  read(20,*)
               enddo
            enddo
         enddo
         do kpt=1,maxk
            do c=1,nbcon(spinflip)
               do v=1,nbval(spinflip)
                 read(20,*) kweight,dum_c,dum_v,all_eig_c(c,kpt,spinflip),all_eig_v(v,kpt,spinflip)
               enddo
            enddo
         enddo
         do kpt=1,koffset
            do c=1,nbcon(fspin)
               do v=1,nbval(fspin)
                  read(21,*)
               enddo
            enddo
         enddo
         do kpt=1,maxk
            do c=1,nbcon(fspin)
               do v=1,nbval(fspin)
                 read(21,*) kweight,dum_c,dum_v,all_eig_c(c,kpt,fspin),all_eig_v(v,kpt,fspin)
               enddo
            enddo
         enddo
         close(20)
         close(21)

         count=0
         do kpt=1,maxk
            do c=1,nbcon(spinflip)
               do v=1,nbval(fspin)
                  if ((all_eig_c(c,kpt,spinflip)-all_eig_v(v,kpt,fspin))<encut_sp) then
                     if ((nehmen_b_sp(v)==1).and.(nehmen_b_sp(nbval(spinflip)+c)==1)) then
                        count=count+1
                     endif
                  endif
               enddo
            enddo
         enddo
         matdim=count
         deallocate(all_eig_c)
         deallocate(all_eig_v)
      endif

      end subroutine get_matdim_from_optics

!     *******************************************************************
!     Get the LDA values and the transition-matrix elements from an
!     input file:
!     kpt_weight is non important so far, c is a conduction band and 
!     v a valence band index. eig_c and eig_v are the appropriate LDA 
!     values to c and v.
!     px_re/px_im is the X, py_re/py_im the Y and pz_re/pz_im the Z value 
!     of the matrix element <ck|p_{x,y,z}|vk>.
!     Here we apply the same cutoff as for the calculation of the
!     excitonic matrix. Therefore, both cutoff energies must be the same!

      subroutine read_optics()
      use excivar
      implicit none
! local
      integer  :: c,v,k,spin,spinp,kpt,kptp
      integer  :: dum_c,dum_v,dim_k_kp,count_ma,count_sp,fspin
      real(mq) :: kweight,eig_c,eig_v,px_re,px_im,py_re,py_im,pz_re,pz_im,dum
      real(mq), pointer :: all_eig_c(:,:,:),all_eig_v(:,:,:),all_kweight(:)

      matdim_ma=0
      nehmen_ma=0
      count_ma=0
      count_sp=0

      nehmen_ma=0
      nehmen=0
      nehmen_loc=0
      cvk_tr_ma=0
      cvk_tr_sp=0
      if (beyondtd==1) then
         cvk_artr_sp=0
         nehmen_loc_ar=0
      endif
      dim_at_k_ma=0
      dim_at_k=0

      if (spinflip==0) then
         do spin=1,ispin
            open(20,file=optics_file(spin),action='read',status='old')
            do kpt=1,maxk
               do c=1,nbcon(spin)
                  do v=1,nbval(spin)
                     read(20,*) kweight,dum_c,dum_v,eig_c,eig_v,px_re,px_im,py_re,py_im,pz_re,pz_im
                     if ((eig_c-eig_v)<encut_sp) then
                        if ((nehmen_b_sp(v)==1).and.(nehmen_b_sp(nbval(spin)+c)==1)) then
                           count_sp=count_sp+1
                           if ((matstart(node_me)<count_sp).and.(count_sp<=matstart(node_me+1))) then
                              nehmen_loc(v,c,kpt,spin)=1
                           endif
                           if (beyondtd==1) then
                              if ((matstart(node_me)<matdim+count_sp).and.(matdim+count_sp<=matstart(node_me+1))) then
                                 nehmen_loc_ar(v,c,kpt,spin)=1
                              endif
                           endif
                           nehmen(v,c,kpt,spin)=1
                           dim_at_k(kpt,spin)=dim_at_k(kpt,spin)+1
                           ! The optical matrix elements in VASP are multiplied by a strange 
                           ! factor inv_imun=(0._mq,-1._mq). Remove this factor.
                           ! Moreover, there is a superfluous conjugation. Remove this as well.
                           optmat(count_sp,1)=imun*conjg(cmplx(px_re,px_im))/(eig_c-eig_v)
                           optmat(count_sp,2)=imun*conjg(cmplx(py_re,py_im))/(eig_c-eig_v)
                           optmat(count_sp,3)=imun*conjg(cmplx(pz_re,pz_im))/(eig_c-eig_v)
                           ldagw(count_sp)=(gw(c+nbval(spin),kpt,spin)+eig_c)-(gw(v,kpt,spin)+eig_v)+qpshiftsci
                           kptweight(count_sp)=kweight
                        endif
                     endif
                     cvk_tr_sp(v,c,kpt,spin)=count_sp
                     if (beyondtd==1) then
                        cvk_artr_sp(v,c,kpt,spin)=matdim+count_sp
                     endif
                     if ((eig_c-eig_v).lt.encut_ma) then
                        if ((nehmen_b_ma(v)==1).and.(nehmen_b_ma(nbval(spin)+c)==1)) then
                           count_ma=count_ma+1
                           nehmen_ma(v,c,kpt,spin)=1
                           dim_at_k_ma(kpt,spin)=dim_at_k_ma(kpt,spin)+1
                           matdim_ma=matdim_ma+1
                        endif
                     endif
                     cvk_tr_ma(v,c,kpt,spin)=count_ma
                  enddo
               enddo
               nehmen_loc(0,0,kpt,spin)=maxval(nehmen_loc(:,:,kpt,spin))
               if (beyondtd==1) then
                  nehmen_loc_ar(0,0,kpt,spin)=maxval(nehmen_loc_ar(:,:,kpt,spin))
               endif
            enddo
            close(20)
         enddo
      elseif (spinflip==1.or.spinflip==2) then
         spin=1
         fspin=3-spinflip
         open(20,file=optics_file(spinflip),action='read',status='old')
         open(21,file=optics_file(fspin),action='read',status='old')
         allocate(all_kweight(maxk))
         allocate(all_eig_c(nbconmax,maxk,ispinall))
         allocate(all_eig_v(nbvalmax,maxk,ispinall))
         all_kweight=0._mq
         all_eig_c=0._mq
         all_eig_v=0._mq
         open(20,file=optics_file(spinflip),action='read',status='old')
         open(21,file=optics_file(fspin),action='read',status='old')
         do kpt=1,koffset
            do c=1,nbcon(spinflip)
               do v=1,nbval(spinflip)
                  read(20,*)
               enddo
            enddo
         enddo
         do kpt=1,maxk
            do c=1,nbcon(spinflip)
               do v=1,nbval(spinflip)
                 read(20,*) all_kweight(kpt),dum_c,dum_v,all_eig_c(c,kpt,spinflip),all_eig_v(v,kpt,spinflip)
               enddo
            enddo
         enddo
         do kpt=1,koffset
            do c=1,nbcon(fspin)
               do v=1,nbval(fspin)
                  read(21,*)
               enddo
            enddo
         enddo
         do kpt=1,maxk
            do c=1,nbcon(fspin)
               do v=1,nbval(fspin)
                 read(21,*) all_kweight(kpt),dum_c,dum_v,all_eig_c(c,kpt,fspin),all_eig_v(v,kpt,fspin)
               enddo
            enddo
         enddo
         close(20)
         close(21)

         do kpt=1,maxk
            do c=1,nbcon(spinflip)
               do v=1,nbval(fspin)
                  if ((all_eig_c(c,kpt,spinflip)-all_eig_v(v,kpt,fspin))<encut_sp) then
                     if ((nehmen_b_sp(v)==1).and.(nehmen_b_sp(nbval(spinflip)+c)==1)) then
                        count_sp=count_sp+1
                        if ((matstart(node_me)<count_sp).and.(count_sp<=matstart(node_me+1))) then
                           nehmen_loc(v,c,kpt,spin)=1
                        endif
                        if (beyondtd==1) then
                           if ((matstart(node_me)<matdim+count_sp).and.(matdim+count_sp<=matstart(node_me+1))) then
                              nehmen_loc_ar(v,c,kpt,spin)=1
                           endif
                        endif
                        nehmen(v,c,kpt,spin)=1
                        dim_at_k(kpt,spin)=dim_at_k(kpt,spin)+1
                        optmat(count_sp,1)=czero
                        optmat(count_sp,2)=czero
                        optmat(count_sp,3)=czero
                        ldagw(count_sp)=(gw(c+nbval(spinflip),kpt,spinflip)+all_eig_c(c,kpt,spinflip)) &
                                       -(gw(v,kpt,fspin)+all_eig_v(v,kpt,fspin))+qpshiftsci
                        kptweight(count_sp)=all_kweight(kpt)
                     endif
                  endif
                  cvk_tr_sp(v,c,kpt,spin)=count_sp
                  if (beyondtd==1) then
                     cvk_artr_sp(v,c,kpt,spin)=matdim+count_sp
                  endif
                  if ((all_eig_c(c,kpt,spinflip)-all_eig_v(v,kpt,fspin)).lt.encut_ma) then
                     if ((nehmen_b_ma(v)==1).and.(nehmen_b_ma(nbval(spinflip)+c)==1)) then
                        count_ma=count_ma+1
                        nehmen_ma(v,c,kpt,spin)=1
                        dim_at_k_ma(kpt,spin)=dim_at_k_ma(kpt,spin)+1
                        matdim_ma=matdim_ma+1
                     endif
                  endif
                  cvk_tr_ma(v,c,kpt,spin)=count_ma
               enddo
            enddo
            nehmen_loc(0,0,kpt,spin)=maxval(nehmen_loc(:,:,kpt,spin))
            if (beyondtd==1) then
               nehmen_loc_ar(0,0,kpt,spin)=maxval(nehmen_loc_ar(:,:,kpt,spin))
            endif
         enddo
         deallocate(all_kweight)
         deallocate(all_eig_c)
         deallocate(all_eig_v)
      endif
    
      end subroutine read_optics

!     *******************************************************************
!     This part of the code simply reads and builds the matrix.
!ff   Ok,ok delete the simply ... but it reads the matrix and its fast.

      subroutine read_and_build_matrix()
      use excivar
      implicit none
! local 
      integer              :: my_id,my_unit,my_unit_last,my_unit_line
      integer              :: swap_threshold,swap_count,reader
      integer              :: spin,spinp,kpt,kptp,c,cp,v,vp,i,j,kptp_start
      integer              :: ikl,ik,vcks_vcksp(8)
      integer              :: file_lindex(0:nthreads_ma-1)
      integer*8            :: nset,count_rv,l,lp,firstread,lastread,read_sets,tmp8
      integer*8            :: first_entry(0:nthreads_ma),firstentry,firstread_old
      integer, pointer     :: file_no(:)
      integer, pointer     :: nehmen_p(:,:,:,:),nehmen_loc_p(:,:,:,:)
      integer, pointer     :: dim_at_k_p(:,:)
      integer, pointer     :: cvk_tr_sp_p(:,:,:,:),cvk_tr_ma_p(:,:,:,:)
      integer, pointer     :: cvk_artr_sp_p(:,:,:,:),nehmen_loc_ar_p(:,:,:,:)
      real(4)              :: imaxk
      real(mq)             :: z1,z2,z3
#ifdef checkbuild
      real(mq)             :: tol,defvalabs
      complex(mq)          :: defval
#endif
      complex(4), pointer  :: swap(:) ! der steht leider nur so in der Datei
      complex(mq), pointer :: bsematrix_p(:,:)
      logical              :: build
      character(255)       :: matrixname

      integer              :: ierr,recv

      imaxk=1._4/real(maxk,kind=4)


      !ff the new matrix is written in pieces of a length of 8kByte == 1024x8 Byte complex numbers
      swap_threshold=1024
      allocate(swap(swap_threshold))

      !ff initialize the matrix this is necessary only for spin-polarized 
      !ff calculations with lfcorr=0
      bsematrix(:,:)=czero

      !ff open all files - I think/know that this is by far better 
      !ff because we only need to go throught the matrix loop once
      !ff and not once for every file 
      first_entry=(int(matdim_ma+1,kind=8)*int(matdim_ma,kind=8))/int(2*nthreads_ma,kind=8)
      infwritef(*,'(1x,(a),i8)') 'Dimension of the matrix (on disk): ', matdim_ma
      infwritef(*,'(1x,(a),i16)') 'Number of elements in the upper triangle of the matrix (on disk):', &
               (int(matdim_ma+1,kind=8)*int(matdim_ma,kind=8))/int(2,kind=8)
      infwrite
      first_entry(0)=0
      file_lindex(0)=0
      allocate(file_no(1:matdim_ma))
      file_no(:)=-999
      infwritef(*,'(1x,(a))') 'Reading matrix in progress ...'

      ! Find out which part of the matrix is contained in which file.
      do my_id=0,nthreads_ma-1
         ! Get the real name for the matrix file. 
         ! We can use up to 1000 processors with this scheme.
         if (spinflip==0) matrixname=matrix_file
         if (spinflip==1) matrixname=matrix_sfl_file(1)
         if (spinflip==2) matrixname=matrix_sfl_file(2)
         i=len_trim(matrixname)+1
         matrixname(i:i)='_'
         j=int(my_id/100)
         matrixname(i+1:i+1)=char(ichar('0')+j)
         j=my_id-(int(my_id/100)*100)
         j=int(j/10)
         matrixname(i+2:i+2)=char(ichar('0')+j)
         j=my_id-(int(my_id/10)*10)
         matrixname(i+3:i+3)=char(ichar('0')+j)
!         infwrite my_id,matrixname
         my_unit=my_id+20

         ! where do the files start ??
         first_entry(my_id+1)=first_entry(my_id+1)+first_entry(my_id)
         do j=1,matdim_ma
            tmp8=int(j,kind=8)
            ! l is the diagonal element in the jth row
            l=(int(matdim_ma+1,kind=8)-tmp8)*(tmp8-1)+((tmp8-1)*tmp8)/2+1
            if (first_entry(my_id+1)<l) then
               first_entry(my_id+1)=l-1 ! last matrix element in each file
               file_lindex(my_id)=tmp8-1 ! index of last row in in file my_id
               exit
            endif
         enddo
         if (first_entry(my_id+1)>=(int(matdim_ma+1,kind=8)*int(matdim_ma,kind=8))/int(2,kind=8)) then 
            first_entry(my_id+1)=(int(matdim_ma+1,kind=8)*int(matdim_ma,kind=8))/int(2,kind=8)
            file_lindex(my_id)=matdim_ma
         endif
         ! Determine the file-unit numbers for each row.
         if (my_id==0) file_no(1:file_lindex(my_id))=my_unit
         if (my_id>0) file_no(file_lindex(my_id-1)+1:file_lindex(my_id))=my_unit
!         infwritef(*,'((a),i2,(a),6(1x,i12))') 'file ',my_id,' starts-stopps at element no. ',& 
!                first_entry(my_id)+1,first_entry(my_id+1),first_entry(my_id+1)-first_entry(my_id),& 
!                (first_entry(my_id+1)-first_entry(my_id)+swap_threshold)/swap_threshold,file_lindex(my_id)
         ! now open + read the file
         open(my_unit,file=matrixname,form='unformatted',action='read',access='direct',recl=swap_threshold*8)
      enddo

!     -------------------------------------------------------------------
      !ff now this is where the magic happens :)) No honestly let me try to explain:
      !ff 1.) we are going through all v,c,k,s and vp,cp,kp,sp combinations that 
      !ff     could occur in the matrix of course we cycle the loop at the earliest 
      !ff     possible chance when the respective transition is excluded by the 
      !ff     spectra cutoff - that is right - the spectra cutoff only - because 
      !ff     it has to be equal or lower than the matrix cutoff anyway
      !ff 2.) further we calculate the position in the matrix (on disk) directly, 
      !ff     so we dont need to stress our brain where cycling will take us
      !ff     ******      xoxoxo     
      !ff     ******       xoxox
      !ff     ------        oxox 
      !ff     ------         oxo
      !ff     ++++++          xo
      !ff     ++++++           x  
      !ff 3.) the full matrix is distributed in memory in blocks as shown above 
      !ff     (3-nodes) and written to disk as the upper triangle to a number 
      !ff     of files alternating with the data set
      !ff 4.) reading from disk is considered the slowest process involved 
      !ff     (sound plausible, no?) therfore we wont read the matrix twice, 
      !ff     but let the reading process broadcast all its swap to the trailing 
      !ff     nodes
      !ff 5.) some finetuning involves setting pointers to some sub-arrays to avoid 
      !ff     cross-array access
      !ff     dont laugh about it, there is a significant effect from some of them

      ! In beyond Tamm-Dancoff mode the resonant and the antiresonant part
      ! of the Hamiltonian are build. (The coupling terms are added later.)
      ! In this case only the first half of the the MPI processes really
      ! reads and builds the matrix, while the rest only builds.

      do reader=0,nodes-1
         vcks_vcksp=0
         dim_at_k_p   => dim_at_k
         nehmen_loc_p => nehmen_loc
         nehmen_p     => nehmen
         cvk_tr_ma_p  => cvk_tr_ma
         cvk_tr_sp_p  => cvk_tr_sp
         if (beyondtd==1) then
            cvk_artr_sp_p  => cvk_artr_sp
            nehmen_loc_ar_p => nehmen_loc_ar
         endif
         bsematrix_p  => bsematrix
         !ff here we start reading --------------------------------------
!        =========================
         if (node_me==reader) then
!        =========================
            call cpu_time(z1)
            nset=0; firstread=0; lastread=0; read_sets=0; my_unit_last=0

            do spin=1,ispin !--------------------------------------------
               vcks_vcksp(4)=spin ! we need the actual coordinates later 

               do kpt=1,maxk !-------------------------------------------
                  vcks_vcksp(3)=kpt
                  if (dim_at_k(kpt,spin)==0) cycle ! there is nothing to do ... nice
                  do c=1,nbcon(spin) !-----------------------------------
                     vcks_vcksp(2)=c
                     do v=1,nbval(spin) !--------------------------------
                        vcks_vcksp(1)=v 
                        if (nehmen_loc(v,c,kpt,spin)==0) cycle ! if we are outside of our block, cycle
                        ik  = cvk_tr_sp(v,c,kpt,spin)          ! this is the row index in the matrix
                        ikl = ik-matstart(node_me)             ! this is the row index in the block located at node_me
                        l   = cvk_tr_ma(v,c,kpt,spin)-1        ! for the calculation of the matrix index
                        ! number of the last matrix element in the previous line of the matrix on disk - l (pronounce el)
                        lp  = l*(int(matdim_ma,kind=8)-l)+(l*int(l+1,kind=8))/int(2,kind=8)-l 

                        ! find the unit we have to read from                  
                        my_unit=file_no(cvk_tr_ma(v,c,kpt,spin))
                        ! switch the file if necessary
                        if (my_unit>my_unit_last) then
                           lastread=0 
                           my_unit_last=my_unit
                           firstentry=first_entry(my_unit-20)
                           ! infwrite 'switching file',my_unit,cvk_tr_ma(v,c,kpt,spin)
                        endif
                     
                        do spinp=spin,ispin !----------------------------
                           vcks_vcksp(8)=spinp
                           kptp_start=kpt
                           if ((lfcorr.eq.0).and.(spin.ne.spinp)) cycle
                           if (spin.ne.spinp) kptp_start=1

                           do kptp=kptp_start,maxk !---------------------
                              vcks_vcksp(7)=kptp
                              if (dim_at_k_p(kptp,spinp)==0) cycle ! there is nothing to do ... nice

                              do cp=1,nbcon(spinp) !---------------------
                                 vcks_vcksp(6)=cp
                                 do vp=1,nbval(spinp) !------------------
                                    vcks_vcksp(5)=vp
                                    if (nehmen_p(vp,cp,kptp,spinp)==0) cycle
                                    if (ik>cvk_tr_sp_p(vp,cp,kptp,spinp)) cycle ! if we are not in the upper triangle, cycle
                                    ! number of the matrix element on disk
                                    l = lp+cvk_tr_ma_p(vp,cp,kptp,spinp)
                                    
                                    ! read new set if necessary (nothing left in swap to build) 
                                    if (l>lastread) then
                                       ! find the set (within the current file) to read from 
                                       nset=(l-firstentry-1)/swap_threshold+1
!                                      infwrite 'reading-a',my_unit,nset,l,lastread,firstentry,cvk_tr_ma(v,c,kpt,spin)
                                       read(my_unit,rec=nset,err=666) swap(1:swap_threshold)
                                       firstread=int(nset-1,kind=8)*int(swap_threshold,kind=8)+firstentry
                                       lastread=firstread+int(swap_threshold,kind=8)
!                                      infwrite 'ok-a',firstread,lastread,l-firstread
!                                      mdwrite 'sendinga',firstread,l
                                       !ff this is just for the progress visualization
                                       read_sets=read_sets+1
                                       if (mod(read_sets,int(10000,kind=8))==0) then
                                          write(*,'(1x,i2)',advance='no') node_me
                                       endif
                                    endif

                                    !------------------------------------
                                    ! resonant transitions in the locally stored part of the upper triangle

                                    bsematrix(ikl,cvk_tr_sp_p(vp,cp,kptp,spinp))=-swap(l-firstread)*imaxk
                                    if (beyondtd==1) then
                                       ! corresponding antiresonant transitions
                                       ! (relevant only for single-processor use, otherwise they are located at another node)
                                       if (nehmen_loc_ar(v,c,kpt,spin)==1) then
                                                bsematrix(cvk_artr_sp(v,c,kpt,spin)-matstart(node_me),cvk_artr_sp_p(vp,cp,kptp,spinp))=conjg(swap(l-firstread))*imaxk
                                       endif
                                    endif
                                    !------------------------------------
                                    !ff resonant transitions in the part of the lower triangle stored locally
                                    if (nehmen_loc_p(vp,cp,kptp,spinp)==1) then
                                       bsematrix_p(cvk_tr_sp_p(vp,cp,kptp,spinp)-matstart(node_me),ik)=-conjg(swap(l-firstread))*imaxk
                                    endif
                                    if (beyondtd==1) then
                                       ! corresponding antiresonant transitions
                                       ! (relevant only for single-processor use, otherwise they are located at another node)
                                       if (nehmen_loc_ar_p(vp,cp,kptp,spinp)==1) then
                                          bsematrix_p(cvk_artr_sp_p(vp,cp,kptp,spinp)-matstart(node_me),cvk_artr_sp(v,c,kpt,spin))=swap(l-firstread)*imaxk
                                       endif
                                    endif
                                    !------------------------------------
                                 enddo ! vp
                              enddo ! cp
                           enddo ! kptp 
                        enddo ! spinp
                     enddo ! v
                  enddo ! c
               enddo ! kpt
            enddo ! spin
            call cpu_time(z2)
            write(*,'(1x,(a),i3,(a),f13.3,(a))') 'Processor',node_me,' finished without error. Time:',z2-z1,' sec'
            goto 667
 665        mdwrite 'ERROR during build - STOP',nset,my_unit
            mdwrite 'i',v,c,kpt,spin
            mdwrite 'j',vp,cp,kptp,spinp
            mdwrite l,firstread,lastread
            mdwrite cvk_tr_ma(v,c,kpt,spin),cvk_tr_ma_p(vp,cp,kptp,spinp),lp
            stop
 666        mdwrite 'ERROR during read - STOP',nset,my_unit
            mdwrite 'i',v,c,kpt,spin
            mdwrite 'j',vp,cp,kptp,spinp
            mdwrite l,firstread,lastread
            mdwrite cvk_tr_ma(v,c,kpt,spin),cvk_tr_ma_p(vp,cp,kptp,spinp),lp
            stop
 667        continue
            do my_id=0,nthreads_ma-1
               my_unit=20+my_id
               close(my_unit)
            enddo
            ! this is just the signal to the other guys, that we are done
            vcks_vcksp(1)=-5

!        ========================
         else ! (node_me/=reader) ---------------------------------------
!        ========================
            ! now here the other guys are spending their time 
            ! they receive matrix elements and coordinates from the reader
            ! to build the lower triangle
            do while (vcks_vcksp(1)>=0)
               ! receive swap and some coordinates from the reader
               if (vcks_vcksp(1)>=0.and.node_me>reader) then
!                  write(*,*) 'recv',vcks_vcksp
                  do spin=vcks_vcksp(4),ispin
                     do kpt=vcks_vcksp(3),maxk
                        do c=vcks_vcksp(2),nbcon(spin)
                           do v=vcks_vcksp(1),nbval(spin)
                              if (nehmen(v,c,kpt,spin)==0) cycle 
                              ! do not build matrix elements from the trailing zeros
                              ik  = cvk_tr_sp(v,c,kpt,spin)
                              ikl = ik-matstart(node_me)
                              l   = cvk_tr_ma(v,c,kpt,spin)-1 ! for the calculation of the matrix index
                              lp  = l*(int(matdim_ma,kind=8)-l)+(l*int(l+1,kind=8))/int(2,kind=8)-l

                              do spinp=vcks_vcksp(8),ispin
                                 if ((lfcorr.eq.0).and.(spin.ne.spinp)) cycle
                                 do kptp=vcks_vcksp(7),maxk
                                    do cp=vcks_vcksp(6),nbcon(spinp)
                                       do vp=vcks_vcksp(5),nbval(spinp)
                                          if (nehmen_p(vp,cp,kptp,spinp)==0) cycle
                                          if (ik>cvk_tr_sp_p(vp,cp,kptp,spinp)) cycle
                                          l = lp+cvk_tr_ma_p(vp,cp,kptp,spinp)
                                          if (l-firstread.gt.swap_threshold) goto 668 !complete set read

                                          !------------------------------
                                          ! in case a read set crosses the boarder between two locally stored
                                          ! parts of the matrix (sets do not necessarily end at the end of a
                                          ! matrix row), appears only if the number of matrix files does not
                                          ! coincide with the number of processes used for the spectrum

                                          ! resonant transitions
                                          if (nehmen_loc(v,c,kpt,spin)==1.and.nehmen_p(vp,cp,kptp,spinp)==1) then
#ifdef checkbuild
                                             if (abs(bsematrix(ikl,cvk_tr_sp_p(vp,cp,kptp,spinp))-defvalabs)<tol) then
#endif
                                                bsematrix(ikl,cvk_tr_sp_p(vp,cp,kptp,spinp))=-swap(l-firstread)*imaxk
#ifdef checkbuild
                                             else
!                                                write(*,*) 'element already read c',reader,node_me, &
!                                                ikl,cvk_tr_sp_p(vp,cp,kptp,spinp), &
!                                                bsematrix(ikl,cvk_tr_sp_p(vp,cp,kptp,spinp))
                                                goto 669
                                             endif
#endif
                                          endif
                                          if (beyondtd==1) then
                                             ! corresponding antiresonant transitions
                                             if (nehmen_loc_ar(v,c,kpt,spin)==1.and.nehmen_p(vp,cp,kptp,spinp)==1) then
#ifdef checkbuild
                                                if (abs(bsematrix(cvk_artr_sp(v,c,kpt,spin)-matstart(node_me),cvk_artr_sp_p(vp,cp,kptp,spinp))-defvalabs)<tol) then
#endif
                                                   bsematrix(cvk_artr_sp(v,c,kpt,spin)-matstart(node_me),cvk_artr_sp_p(vp,cp,kptp,spinp))=conjg(swap(l-firstread))*imaxk
#ifdef checkbuild
                                                else
!                                                   write(*,*) 'element already read c',reader,node_me, &
!                                                   cvk_artr_sp(v,c,kpt,spin)-matstart(node_me),cvk_artr_sp_p(vp,cp,kptp,spinp), &
!                                                   bsematrix(cvk_artr_sp(v,c,kpt,spin)-matstart(node_me),cvk_artr_sp_p(vp,cp,kptp,spinp))
                                                   goto 669
                                                endif
#endif
                                             endif
                                          endif
                                          !------------------------------
                                          ! building the parts of the lower triangle located at nodes>reader

                                          ! resonant transitions
                                          if (nehmen_loc_p(vp,cp,kptp,spinp)==1) then
#ifdef checkbuild
                                             ! do not rewrite the main diagonal
                                             if (cvk_tr_sp(v,c,kpt,spin).ne.cvk_tr_sp_p(vp,cp,kptp,spinp)) then
                                                if (abs(bsematrix_p(cvk_tr_sp_p(vp,cp,kptp,spinp)-matstart(node_me),ik)-defvalabs)<tol) then
#endif
                                                   bsematrix_p(cvk_tr_sp_p(vp,cp,kptp,spinp)-matstart(node_me),ik)=-conjg(swap(l-firstread))*imaxk
#ifdef checkbuild
                                                else
!                                                   write(*,*) 'element already read d',reader,node_me, &
!                                                   cvk_tr_sp_p(vp,cp,kptp,spinp)-matstart(node_me),ik, &
!                                                   bsematrix_p(cvk_tr_sp_p(vp,cp,kptp,spinp)-matstart(node_me),ik)
                                                   goto 669
                                                endif
                                             endif
#endif
                                          endif
                                          if (beyondtd==1) then
                                             ! antiresonant transitions
                                             if (nehmen_loc_ar_p(vp,cp,kptp,spinp)==1) then
#ifdef checkbuild
                                                ! do not rewrite the main diagonal
                                                if (cvk_artr_sp(v,c,kpt,spin).ne.cvk_artr_sp_p(vp,cp,kptp,spinp)) then
                                                   if (abs(bsematrix_p(cvk_artr_sp_p(vp,cp,kptp,spinp)-matstart(node_me),matdim+ik)-defvalabs)<tol) then
#endif
                                                      bsematrix_p(cvk_artr_sp_p(vp,cp,kptp,spinp)-matstart(node_me),cvk_artr_sp(v,c,kpt,spin))=swap(l-firstread)*imaxk
#ifdef checkbuild
                                                   else
!                                                      write(*,*) 'element already read d',reader,node_me, &
!                                                      cvk_artr_sp_p(vp,cp,kptp,spinp)-matstart(node_me),cvk_artr_sp(v,c,kpt,spin), &
!                                                      bsematrix_p(cvk_artr_sp_p(vp,cp,kptp,spinp)-matstart(node_me),cvk_artr_sp(v,c,kpt,spin))
                                                      goto 669
                                                   endif
                                                endif
#endif
                                             endif
                                          endif
                                          !------------------------------
                                       enddo ! vp
                                       vcks_vcksp(5)=1
                                    enddo ! cp
                                    vcks_vcksp(6)=1
                                 enddo ! kptp
                                 vcks_vcksp(7)=kpt !ff Im sure
                                 ! we need the full rectangle only for the upper right square spin-spinp
                                 ! but it is easier like that
!CR                                 if ((spin.eq.spinp).and.(ispin==2)) vcks_vcksp(7)=1 ! (I think this is wrong in general.)
                                 if (ispin==2) vcks_vcksp(7)=1 
                              enddo ! spinp
                              vcks_vcksp(8)=spin
                           enddo ! v
                           vcks_vcksp(1)=1
                        enddo ! c
                        vcks_vcksp(2)=1
                     enddo ! kpt
                     vcks_vcksp(3)=1
                  enddo !spin
               endif ! if receiving
 668           continue
#ifdef checkbuild
               firstread_old=firstread
#endif
            enddo ! while receiving
         endif
      enddo !reader
      goto 670
 669  mdwrite 'ERROR during build - STOP'
      stop
 670  continue
 
!     -------------------------------------------------------------------
      ! Patch the main diagonal with the RPA transition energies.
      do i=1,matdim_node(node_me)
         j=i+matstart(node_me)
         if (beyondtd==0) then
            bsematrix(i,j)=ldagw(j)+bsematrix(i,j)
         elseif (beyondtd==1) then
            if (j<=matdim) then
               bsematrix(i,j)=ldagw(j)+bsematrix(i,j)
            else
               bsematrix(i,j)=-ldagw(j-matdim)+bsematrix(i,j)
            endif
         endif
      enddo
      call cpu_time(z3)

!     -------------------------------------------------------------------
      infwritef(*,'(1x,(a))') 'Reading matrix finished.'
      infwrite
      infwritef(*,'(1x,(a),f13.3,(a))') 'Time for reading and building: ',z3-z1,' sec'
      infwrite



!#define debug
#ifdef debug
      if (beyondtd==1) then
         call sleep(2)
         call sleep(node_me*2)
         do spin=1,ispin
            do kpt=1,MaxK
               do c=1,nbcon(spin)
                  do v=1,nbval(spin)
                     if (nehmen_loc(v,c,kpt,spin)==1.or.nehmen_loc_ar(v,c,kpt,spin)==1) then
                        write(*,'(8(1x,i5))') node_me,v,c,kpt,spin,nehmen_loc(v,c,kpt,spin),nehmen_loc_ar(v,c,kpt,spin)
                     endif
                  enddo
               enddo
            enddo
         enddo
         call sleep(2)
      endif
#endif

!#define debug
#ifdef debug
      call sleep(2)
      if (matdim.lt.20) then
         call sleep(node_me*2)
         mdwrite 'bsematrix:'
         do i=1,matdim_node(node_me)
            if (beyondtd==0) then
               write(*,'(2x,10(1x,e13.6))') (abs(bsematrix(i,j)),j=1,matdim)
            elseif (beyondtd==1) then
               write(*,'(10(1x,e13.6))') (abs(bsematrix(i,j)),j=1,matdim_btd)
            endif
         enddo
         call sleep(2)
      endif
      call sleep(2)
#endif

      deallocate(swap)
      deallocate(file_no)
      nullify(nehmen_p)
      nullify(nehmen_loc_p)
      if (beyondtd==1) nullify(nehmen_loc_ar_p)
      nullify(dim_at_k_p)
      nullify(cvk_tr_ma_p)
      nullify(cvk_tr_sp_p)
      if (beyondtd==1) nullify(cvk_artr_sp_p)
      nullify(bsematrix_p)

      end subroutine read_and_build_matrix

!     *******************************************************************


!     *******************************************************************
!     Check whether the Hamiltonian is Hermitian.

      subroutine check_hermiticity(matrix)
      use excivar
      implicit none
      complex(mq), pointer :: matrix(:,:)
!local
      integer              :: il,i,n,dim
      real(mq), pointer    :: re_diff(:),im_diff(:)
      complex(mq), pointer :: v1(:),v2(:)

      real(mq), parameter  :: tol=1.e-10_mq

      integer              :: ierr

      dim=size(matrix,2)
      allocate(re_diff(dim))
      allocate(im_diff(dim))
      allocate(v1(dim))
      allocate(v2(dim))

      do n=0,nodes-1
         do il=1,matdim_node(n)
            i=matstart(n)+il
            if (n==node_me) v1(:)=matrix(il,:)
            v2(:)=conjg(matrix(:,i))
            if (n==node_me) then
               re_diff(:)=abs(real(v1(:)-v2(:),mq))
               im_diff(:)=abs(real(aimag(v1(:)-v2(:)),mq))
               if (maxval(re_diff)>tol.or.maxval(im_diff)>tol) then
                  write(*,'(1x,(a),e13.6,(a),i8)') 'Violation of Hermiticity within ',tol,' found in line ',i
                  write(*,'(1x,(a),2(1x,e13.6))') 'Maximum difference (real and imaginary parts): ',maxval(re_diff),maxval(im_diff)
                  stop
               endif
            endif
         enddo
      enddo
      infwritef(*,'(1x,(a))') 'Matrix is Hermitian.'
      infwrite

      deallocate(re_diff)
      deallocate(im_diff)
      deallocate(v1)
      deallocate(v2)

      end subroutine check_hermiticity

!     *******************************************************************
!     Calculate f*H with f = diag(1,...,1,-1,...,-1) what should result
!     in a positive definite definite Hermitian matrix.

      subroutine transform_to_pos_def_matrix()
      use excivar
      implicit none
!local
      integer :: i

      integer :: ierr
    
      do i=1,matdim_node(node_me)
         if (matstart(node_me)+i.le.matdim) cycle
         bsematrix(i,:)=-bsematrix(i,:)
      enddo


      infwritef(*,'(1x,(a))') 'Transformed to positive definite matrix.'
      infwrite

      call check_hermiticity(bsematrix)

      end subroutine transform_to_pos_def_matrix

!************************************************************************
!     This is the main routine, in fact.
!************************************************************************

      subroutine get_exciton_spectra()
      use excivar
      implicit none

! local
      integer              :: dir,ierror,i,j,ierr,iec,matdim_
      integer, pointer     :: nbval_tmp(:),nbcon_tmp(:)
      complex(mq), pointer :: mat_col(:),opt_col(:)

      node_lead=0; node_me=0; nodes=1; ierr=0

!     -------------------------------------------------------------------
!     Nullify pointer.

      nullify(nehmen,nehmen_ma)
      nullify(dim_at_k,dim_at_k_ma) 
      nullify(ldagw,gw,bse_eval) 
      nullify(kptweight)
      nullify(bsematrix,optmat) 
      nullify(mat_col,opt_col)

!     -------------------------------------------------------------------
!     Read input.
      call read_parameters()
!      call write_parameter_list()

!     -------------------------------------------------------------------
!     Allocate fields. 

      nbands=nbval(1)+nbcon(1)
      nbvalmax=maxval(nbval)
      nbconmax=maxval(nbcon)

      allocate(gw(nbands,maxk,ispinall),stat=ierror)
      allocate(nehmen(nbvalmax,nbconmax,maxk,ispin),stat=ierror)       
      allocate(nehmen_ma(nbvalmax,nbconmax,maxk,ispin),stat=ierror)
      allocate(nehmen_loc(0:nbvalmax,0:nbconmax,maxk,ispin),stat=ierror)
      if (beyondtd==1) allocate(nehmen_loc_ar(0:nbvalmax,0:nbconmax,maxk,ispin),stat=ierror)
      allocate(nehmen_b_ma(nbands),stat=ierror)
      allocate(nehmen_b_sp(nbands),stat=ierror)
      allocate(dim_at_k(maxk,ispin),stat=ierror)
      allocate(dim_at_k_ma(maxk,ispin),stat=ierror)
      allocate(cvk_tr_ma(nbvalmax,nbconmax,maxk,ispin),stat=ierror)
      allocate(cvk_tr_sp(nbvalmax,nbconmax,maxk,ispin),stat=ierror)
      if (beyondtd==1) allocate(cvk_artr_sp(nbvalmax,nbconmax,maxk,ispin),stat=ierror)

      infwritef(*,'(1x,(a))') 'Building the matrix:'
      infwritef(*,'(1x,(a))') '--------------------'

!     -------------------------------------------------------------------
!     Read QP shifts.

      infwritef(*,'(1x,(a))') 'QP shifts are read.'
      infwrite

!     -------------------------------------------------------------------
!     Determine the dimension of the matrix.

      call get_matdim_from_optics()
      if (beyondtd==0) then
         infwritef(*,'(1x,(a),i8)') 'Dimension of the matrix: ',matdim
         infwritef(*,'(1x,(a),i16)') 'Number of elements in the whole matrix:', &
                                     int(matdim,kind=8)**2
      elseif (beyondtd==1) then
         nevcg=2*nevcg
         matdim_btd=2*matdim
         infwritef(*,'(1x,(a),i8)') 'Dimension of the matrix: ',matdim_btd
         infwritef(*,'(1x,(a),i16)') 'Number of elements in the whole matrix:', &
                                     int(matdim_btd,kind=8)**2
      endif
      infwrite

      matstart(0:255)=0
      if (beyondtd==0) then
         do i=0,nodes-1
            matdim_node(i)=matdim/nodes
            if (i.lt.mod(matdim,nodes)) matdim_node(i)=matdim_node(i)+1
            matstart(i+1)=matstart(i)+matdim_node(i)
         enddo
         j=max(nodes-1,1)
!        Write output only for first and last node.
         do i=0,nodes-1,j
            do_io write(*,'(1x,(a),i3,(a),i7,(a),i7,3x,(a),i7,i7)') &
                        'Dimension of the matrix at processor ',i,': ',&
                        matdim_node(i),' x ',matdim,'starts/ends at ',matstart(i)+1,matstart(i+1)
         enddo
         infwrite
!        Memory estimate (complex number = 8B, only the matrix itself).
#ifndef prec_double
         infwritef(*,'(1x,(a),f13.3,(a))') 'Memory requirements: ', &
              (real(matdim,mq)**2*8._mq)/real(1024**3,mq),' GB'
#else
         infwritef(*,'(1x,(a),f13.3,(a))') 'Memory requirements: ', &
              (real(matdim,mq)**2*16._mq)/real(1024**3,mq),' GB'   
#endif
      elseif (beyondtd==1) then
         do i=0,nodes-1
            matdim_node(i)=matdim_btd/nodes
            if (i.lt.mod(matdim_btd,nodes)) matdim_node(i)=matdim_node(i)+1
            matstart(i+1)=matstart(i)+matdim_node(i)
         enddo
         j=max(nodes-1,1)
!        Write output only for first and last node.
         do i=0,nodes-1,j
            do_io write(*,'(1x,(a),i3,(a),i7,(a),i7,3x,(a),i7,i7)') &
                        'Dimension of the matrix at processor ',i,': ',&
                        matdim_node(i),' x ',matdim_btd,'starts/ends at ',matstart(i)+1,matstart(i+1)
         enddo
         infwrite
!        Memory estimate (complex number = 8B, only the matrix itself).
#ifndef prec_double
         infwritef(*,'(1x,(a),f13.3,(a))') 'Memory requirements: ', &
              (real(matdim_btd,mq)**2*8._mq)/real(1024**3,mq),' GB'
#else
         infwritef(*,'(1x,(a),f13.3,(a))') 'Memory requirements: ', &
              (real(matdim_btd,mq)**2*16._mq)/real(1024**3,mq),' GB'
#endif
      endif
      infwrite

!     -------------------------------------------------------------------
!     Allocate fields.
   
      allocate(optmat(matdim,3))
      allocate(ldagw(matdim))  
      allocate(kptweight(matdim))
      if (beyondtd==0) then
         allocate(bse_eval(matdim))
      elseif (beyondtd==1) then
         allocate(bse_eval(matdim_btd))
      endif

!     -------------------------------------------------------------------
!     Get the LDA values and the transition-matrix elements from an
!     input file.

      call read_optics()
      infwritef(*,'(1x,(a))') 'Optical transition matrix elements are read.'
      infwrite

!     -------------------------------------------------------------------
!     For spin flips the rest of the code should look like an unpolarized
!     calculation.

      if (spinflip==1.or.spinflip==2) then
         allocate(nbcon_tmp(ispinall))
         allocate(nbval_tmp(ispinall))
         nbcon_tmp=nbcon
         nbval_tmp=nbval
         deallocate(nbcon)
         deallocate(nbval)
         allocate(nbcon(ispin))
         allocate(nbval(ispin))
         if (spinflip==1) then
            nbcon(ispin)=nbcon_tmp(1)
            nbval(ispin)=nbval_tmp(2)
         elseif (spinflip==2) then
            nbcon(ispin)=nbcon_tmp(2)
            nbval(ispin)=nbval_tmp(1)
         endif
         deallocate(nbcon_tmp)
         deallocate(nbval_tmp)
      endif

!     -------------------------------------------------------------------
!     The mode of calculation is chosen here.

!     ===================================================================
      if (exciton.eq.1) then 
!     ===================================================================
!        Allocate, read, and build the excitonic matrix.
         if (beyondtd==0) then
            allocate(bsematrix(matdim_node(node_me),matdim),stat=ierror)
         elseif (beyondtd==1) then
            allocate(bsematrix(matdim_node(node_me),matdim_btd),stat=ierror)
         endif
         if (ierror==0) then
            infwritef(*,'(1x,(a))') 'Memory for matrix allocated.'
         else
            infwritef(*,'(1x,(a))') 'No more memory available. Stopping execution.'
            stop
         endif

         call read_and_build_matrix()
         if (checkposdef==1) call transform_to_pos_def_matrix()

         call write2file(bsematrix, matdim, matdim_node(node_me))

         deallocate(cvk_tr_ma,cvk_tr_sp)
         if (beyondtd==1) deallocate(cvk_artr_sp)
         deallocate(dim_at_k,dim_at_k_ma)

!        ----------------------------------------------------------------
!        Rescale the optical matrixelements to account for the 
!        generalized EV problem.

!     ===================================================================
      elseif (exciton==4) then       ! visualize A_lambda over k
!     ===================================================================         
!     ===================================================================
      elseif (exciton==0) then       ! single particle stuff follows
!     ===================================================================
!     ===================================================================
      endif

      deallocate(nehmen,nehmen_ma,nehmen_loc)
      if (beyondtd==1) deallocate(nehmen_loc_ar)
      deallocate(nehmen_b_ma,nehmen_b_sp)
      deallocate(gw,ldagw)
      deallocate(kptweight)
      deallocate(optmat)
      if (exciton.eq.1) then
         deallocate(bsematrix)
      endif
      deallocate(bse_eval)
      deallocate(nbval)
      deallocate(nbcon)

 100  end subroutine get_exciton_spectra

      end module excihelper

!************************************************************************
!************************************************************************

      program get_exciton
      use excivar
      use excihelper
      implicit none
 
      call get_exciton_spectra()

      end program get_exciton
