program bidiag_gen
  implicit none
  integer, parameter :: dp = selected_real_kind(6)
  integer :: M, N, lda, ldu, ldvt, info, i, j, K
  real(dp), allocatable :: A(:,:), B(:,:), U(:,:), VT(:,:)
  real(dp), allocatable :: D(:), E(:), tauq(:), taup(:), work(:)
  character(len=100) :: line
  integer :: lwork
  open(unit=10, file="../args.txt", status='old')
  read(10,*) M, N
  close(10)

  lda = M
  ldu = M
  ldvt = N
  K = min(M, N)

  allocate(A(M,N), B(M,N), U(M,M), VT(N,N))
  allocate(D(K), E(K-1), tauq(K), taup(K))
  lwork = max(1, 5*max(M,N))
  allocate(work(lwork))

  call random_seed()
  call random_number(A)
  A = 20.0_dp*A - 10.0_dp

  ! 保存 A
  open(20, file="../input/A_gm.bin", form="unformatted", access="stream")
  write(20) A
  close(20)

  call sgebrd(M, N, A, lda, D, E, tauq, taup, work, lwork, info)

  ! 构造双对角矩阵 B
  B = 0.0_dp
  do i = 1, K
     B(i,i) = D(i)
     if (i < K) B(i,i+1) = E(i)
  end do

  ! 生成 U
  U = A
  call sorgbr('Q', M, M, N, U, ldu, tauq, work, lwork, info)

  ! 生成 Vt
  VT = A(1:N, 1:N)
  call sorgbr('P', N, N, M, VT, ldvt, taup, work, lwork, info)

  ! 保存结果
  open(21, file="../output/U_golden.bin", form="unformatted", access="stream")
  write(21) U
  close(21)

  open(22, file="../output/B_golden.bin", form="unformatted", access="stream")
  write(22) B
  close(22)

  open(23, file="../output/Vt_golden.bin", form="unformatted", access="stream")
  write(23) VT
  close(23)

end program bidiag_gen
