program bidiag_gen
  implicit none
  ! 使用单精度，与 LAPACK 的 s* 函数匹配
  integer, parameter :: sp = selected_real_kind(6)
  integer :: M, N, lda, ldu, ldvt, info, i, j, K
  real(sp), allocatable :: A(:,:), Acopy(:,:), B(:,:), U(:,:), VT(:,:)
  real(sp), allocatable :: D(:), E(:), tauq(:), taup(:), work(:)
  real(sp) :: dummy(1)
  character(len=100) :: line
  integer :: lwork, query_lwork, ierr
  logical :: alloc_ok
  real(sp) :: start_time, end_time

  ! 读取矩阵维度
  open(unit=10, file="../args.txt", status='old', iostat=ierr)
  if (ierr /= 0) then
     print *, "Error opening args.txt"
     stop
  endif
  read(10,*,iostat=ierr) M, N
  if (ierr /= 0) then
     print *, "Error reading dimensions from args.txt"
     stop
  endif
  close(10)
  print *, "Matrix dimensions: M=", M, " N=", N

  ! 设置各种维度
  lda = M
  ldu = M
  ldvt = N
  K = min(M, N)
  print *, "K=", K

  ! 分配主要数组
  print *, "Allocating arrays..."
  alloc_ok = .true.
  allocate(A(M,N), stat=ierr)
  if (ierr /= 0) then
     print *, "Failed to allocate A"; alloc_ok = .false.
  endif
  allocate(Acopy(M,N), stat=ierr)
  if (ierr /= 0) then
     print *, "Failed to allocate Acopy"; alloc_ok = .false.
  endif
  allocate(B(M,N), stat=ierr)
  if (ierr /= 0) then
     print *, "Failed to allocate B"; alloc_ok = .false.
  endif
  allocate(U(M,M), stat=ierr)
  if (ierr /= 0) then
     print *, "Failed to allocate U"; alloc_ok = .false.
  endif
  allocate(VT(N,N), stat=ierr)
  if (ierr /= 0) then
     print *, "Failed to allocate VT"; alloc_ok = .false.
  endif
  allocate(D(K), stat=ierr)
  if (ierr /= 0) then
     print *, "Failed to allocate D"; alloc_ok = .false.
  endif
  allocate(E(K-1), stat=ierr)
  if (ierr /= 0) then
     print *, "Failed to allocate E"; alloc_ok = .false.
  endif
  allocate(tauq(K), stat=ierr)
  if (ierr /= 0) then
     print *, "Failed to allocate tauq"; alloc_ok = .false.
  endif
  allocate(taup(K), stat=ierr)
  if (ierr /= 0) then
     print *, "Failed to allocate taup"; alloc_ok = .false.
  endif

  if (.not. alloc_ok) then
     print *, "Memory allocation failed"
     stop
  endif
  print *, "All arrays allocated successfully"

  ! 生成随机矩阵 A
  print *, "Generating random matrix..."
  call random_seed()
  call random_number(A)
  A = 20.0_sp*A - 10.0_sp
  Acopy = A  ! 保存原始 A 的副本

  ! 保存原始矩阵 A
  print *, "Saving original matrix A..."
  open(20, file="../input/A_gm.bin", form="unformatted", access="stream", iostat=ierr)
  if (ierr /= 0) then
     print *, "Error opening A_gm.bin for writing"
     stop
  endif
  write(20) A
  close(20)

  ! 查询最优工作空间大小
  print *, "Querying optimal workspace size..."
  lwork = -1
  call sgebrd(M, N, A, lda, D, E, tauq, taup, dummy, lwork, info)
  if (info /= 0) then
     print *, "Workspace query failed with info = ", info
     stop
  endif
  query_lwork = int(dummy(1))
  print *, "Optimal workspace size = ", query_lwork

  ! 为 GEBRD 和 ORGBR 分配足够大的工作空间
  lwork = max(query_lwork, 5*max(M,N))
  print *, "Allocating workspace of size ", lwork
  allocate(work(lwork), stat=ierr)
  if (ierr /= 0) then
     print *, "Failed to allocate workspace"
     stop
  endif

  ! 执行双对角化 + U/Vt生成计时
  print *, "Computing bidiagonal form and orthogonal matrices..."
  call cpu_time(start_time)

  call sgebrd(M, N, A, lda, D, E, tauq, taup, work, lwork, info)
  if (info /= 0) then
     print *, "sgebrd failed with info = ", info
     stop
  endif
  print *, "Bidiagonal form computed successfully"

  ! 构造双对角矩阵 B
  print *, "Constructing bidiagonal matrix B..."
  B = 0.0_sp
  do i = 1, K
     B(i,i) = D(i)
     if (i < K) B(i,i+1) = E(i)
  end do

  ! 生成 U
  print *, "Generating U..."
  U = 0.0_sp
  U(1:M,1:N) = A
  call sorgbr('Q', M, M, N, U, ldu, tauq, work, lwork, info)
  if (info /= 0) then
     print *, "sorgbr(Q) failed with info = ", info
     stop
  endif
  print *, "U generated successfully"

  ! 生成 Vt
  print *, "Generating Vt..."
  VT(1:N,1:N) = A(1:N,1:N)
  call sorgbr('P', N, N, M, VT, ldvt, taup, work, lwork, info)
  if (info /= 0) then
     print *, "sorgbr(P) failed with info = ", info
     stop
  endif
  print *, "Vt generated successfully"

  call cpu_time(end_time)
  print *, "Time for bidiagonal + orthogonal matrices: ", end_time - start_time, " seconds"

  ! 保存结果
  print *, "Saving results..."
  open(21, file="../output/U_golden.bin", form="unformatted", access="stream", iostat=ierr)
  if (ierr == 0) then
    write(21) U
    close(21)
  else
    print *, "Error saving U_golden.bin"
    stop
  endif

  open(22, file="../output/B_golden.bin", form="unformatted", access="stream", iostat=ierr)
  if (ierr == 0) then
    write(22) B
    close(22)
  else
    print *, "Error saving B_golden.bin"
    stop
  endif

  open(23, file="../output/Vt_golden.bin", form="unformatted", access="stream", iostat=ierr)
  if (ierr == 0) then
    write(23) VT
    close(23)
  else
    print *, "Error saving Vt_golden.bin"
    stop
  endif

  ! 验证结果
  print *, "Verifying results..."
  ! 创建临时矩阵用于验证
  block
    real(sp), allocatable :: temp(:,:), result(:,:)
    allocate(temp(M,N), result(M,N))
    
    ! U(M,M) × B(M,N) -> temp(M,N)
    temp = matmul(U(:,1:N), B(1:N,1:N))
    ! temp(M,N) × VT(N,N) -> result(M,N)
    result = matmul(temp, VT)
    
    ! 计算误差
    result = result - Acopy
    print *, "Maximum reconstruction error: ", maxval(abs(result))
    
    deallocate(temp, result)
  end block

  ! 清理
  print *, "Cleaning up..."
  deallocate(A, Acopy, B, U, VT)
  deallocate(D, E, tauq, taup, work)
  print *, "Program completed successfully"

end program bidiag_gen
