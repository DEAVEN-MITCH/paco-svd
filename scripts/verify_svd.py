#!/usr/bin/python3
import sys
import numpy as np
import os

# 设置误差容限
relative_tol = 1e-3
absolute_tol = 1e-5
error_tol = 1e-3

def load_matrix(filename, shape):
    """加载二进制文件并重塑为指定形状的矩阵"""
    try:
        dtype = np.float32
        expected_elements = np.prod(shape)
        expected_bytes = expected_elements * np.dtype(dtype).itemsize

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        file_size = os.path.getsize(filename)
        if file_size != expected_bytes:
            print(f"[WARNING] File size mismatch for {os.path.basename(filename)}. Expected {expected_bytes} bytes, but got {file_size} bytes.")

        with open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=dtype, count=expected_elements)
            if data.size != expected_elements:
                raise ValueError(f"Error loading {filename}: Expected {expected_elements} elements, but got {data.size} elements.")
            
        return data.reshape(shape)
    except Exception as e:
        print(f"[ERROR] Could not read file {filename}: {e}")
        sys.exit(1)

def verify_orthogonal(matrix, name):
    """验证矩阵是否正交"""
    rows, cols = matrix.shape
    if rows != cols:
        print(f"[WARNING] {name} is not a square matrix ({rows}x{cols})")
        return False

    I = np.eye(rows, dtype=np.float32)
    product = matrix @ matrix.T
    is_orthogonal = np.allclose(product, I, rtol=relative_tol, atol=absolute_tol)
    
    if not is_orthogonal:
        print(f"[WARNING] {name} is not orthogonal")
        diff = np.abs(product - I)
        print(f"Max deviation from identity: {np.max(diff):.6e}")
        print(f"Avg deviation: {np.mean(diff):.6e}")
    else:
        print(f"[INFO] {name} orthogonality verified.")
    return is_orthogonal

def verify_singular_values(sigma, sigma_ref):
    """验证奇异值的准确性"""
    is_accurate = np.allclose(sigma, sigma_ref, rtol=relative_tol, atol=absolute_tol)
    if not is_accurate:
        print("[WARNING] Singular values differ from reference")
        diff = np.abs(sigma - sigma_ref)
        print(f"Max deviation: {np.max(diff):.6e}")
        print(f"Avg deviation: {np.mean(diff):.6e}")
    else:
        print("[INFO] Singular values verified.")
    return is_accurate

def verify_svd_result(output_dir, M, N):
    """验证SVD结果"""
    print(f"\nVerifying SVD results for M={M}, N={N}")
    
    # 加载原始矩阵
    a_path = os.path.join(output_dir, "../input/A_gm.bin")
    A = load_matrix(a_path, (M, N))
    
    # 加载我们的SVD结果
    u_path = os.path.join(output_dir, "U.bin")
    vt_path = os.path.join(output_dir, "Vt.bin")
    sigma_path = os.path.join(output_dir, "sigma.bin")
    
    U = load_matrix(u_path, (M, M))
    Vt = load_matrix(vt_path, (N, N))
    s = load_matrix(sigma_path, (N,))  # 读取为向量
    
    # 构造对角矩阵用于重构
    Sigma = np.zeros((M, N), dtype=np.float32)
    np.fill_diagonal(Sigma, s)
    
    # 计算NumPy的参考结果
    U_ref, s_ref, Vt_ref = np.linalg.svd(A, full_matrices=True)
    
    # 1. 验证U和Vt的正交性
    print("\n1. Verifying Orthogonality...")
    u_ortho = verify_orthogonal(U, "U")
    vt_ortho = verify_orthogonal(Vt, "Vt")
    
    # 2. 验证奇异值
    print("\n2. Verifying Singular Values...")
    s_accurate = verify_singular_values(s, s_ref)  # 直接比较两个向量
    
    # 3. 验证重构精度
    print("\n3. Verifying Reconstruction...")
    A_rec = U @ Sigma @ Vt
    reconstruction_error = np.max(np.abs(A_rec - A))
    reconstruction_passed = reconstruction_error <= error_tol
    print(f"Maximum reconstruction error: {reconstruction_error:.6e}")
    
    # 打印一些额外的调试信息
    print("\n--- Debug Information ---")
    print(f"Singular values (ours): {s[:min(5,N)]}")
    print(f"Singular values (ref):  {s_ref[:min(5,N)]}")
    print(f"U: {U}")
    print(f"Vt: {Vt}")
    print(f"A: {A}")
    print(f"A_rec: {A_rec}")
    # 总结
    print("\n--- Verification Summary ---")
    print(f"1. U Orthogonality:          {'PASSED' if u_ortho else 'FAILED'}")
    print(f"2. Vt Orthogonality:         {'PASSED' if vt_ortho else 'FAILED'}")
    print(f"3. Singular Values Accuracy:  {'PASSED' if s_accurate else 'FAILED'}")
    print(f"4. Reconstruction Accuracy:   {'PASSED' if reconstruction_passed else 'FAILED'}")
    
    is_success = u_ortho and vt_ortho and s_accurate and reconstruction_passed
    
    print("\n--- Overall Result ---")
    if is_success:
        print("✅ SVD Test PASSED!")
    else:
        print("❌ SVD Test FAILED!")
    
    return is_success

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <output_directory>")
        print("  <output_directory>: Directory containing U.bin, sigma.bin, Vt.bin to test.")
        sys.exit(1)

    output_dir = sys.argv[1]
    
    # 读取矩阵维度
    args_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'args.txt')
    try:
        with open(args_file, 'r') as f:
            M, N = map(int, f.readline().split())
    except Exception as e:
        print(f"[ERROR] Failed to read matrix dimensions from args.txt: {e}")
        sys.exit(1)
    
    # 运行验证
    verification_passed = verify_svd_result(output_dir, M, N)
    sys.exit(0 if verification_passed else 1) 