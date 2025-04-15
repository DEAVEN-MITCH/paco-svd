#!/usr/bin/python3
import sys
import numpy as np

# 设置误差容限
relative_tol = 1e-3
absolute_tol = 1e-5
error_tol = 1e-3

def load_matrix(filename, shape):
    """加载二进制文件并重塑为指定形状的矩阵"""
    return np.fromfile(filename, dtype=np.float32).reshape(shape)

def verify_orthogonal(matrix, name):
    """验证矩阵是否正交"""
    I = np.eye(matrix.shape[0], dtype=np.float32)
    product = matrix @ matrix.T
    is_orthogonal = np.allclose(product, I, rtol=relative_tol, atol=absolute_tol)
    if not is_orthogonal:
        print(f"[WARNING] {name} is not orthogonal")
        diff = np.abs(product - I)
        max_diff = np.max(diff)
        print(f"Maximum deviation from identity: {max_diff}")
    return is_orthogonal

def verify_bidiagonal(B, upper=True):
    """验证矩阵是否为上双对角矩阵"""
    M, N = B.shape
    for i in range(M):
        for j in range(N):
            if upper:
                if j < i or j > i + 1:
                    if abs(B[i,j]) > absolute_tol:
                        print(f"[WARNING] Non-zero element at ({i},{j}): {B[i,j]}")
                        return False
            else:
                if j > i or j < i - 1:
                    if abs(B[i,j]) > absolute_tol:
                        print(f"[WARNING] Non-zero element at ({i},{j}): {B[i,j]}")
                        return False
    return True

def verify_result(output_dir, golden_dir, M, N):
    """验证双对角化结果"""
    # 加载输出结果
    U_out = load_matrix(f"{output_dir}/U.bin", (M, M))
    B_out = load_matrix(f"{output_dir}/B.bin", (M, N))
    Vt_out = load_matrix(f"{output_dir}/Vt.bin", (N, N))
    
    # 加载golden结果
    U_golden = load_matrix(f"{golden_dir}/U_golden.bin", (M, M))
    B_golden = load_matrix(f"{golden_dir}/B_golden.bin", (M, N))
    Vt_golden = load_matrix(f"{golden_dir}/Vt_golden.bin", (N, N))
    A_original = load_matrix(f"{golden_dir}/../input/A_gm.bin", (M, N))
    
    # 验证正交性
    u_ortho = verify_orthogonal(U_out, "U")
    v_ortho = verify_orthogonal(Vt_out.T, "V")
    
    # 验证双对角形式
    b_form = verify_bidiagonal(B_out, upper=True)
    
    # 验证重构误差
    A_reconstructed = U_out @ B_out @ Vt_out
    diff = np.abs(A_reconstructed - A_original)
    max_error = np.max(diff)
    mean_error = np.mean(diff)
    
    print(f"Maximum reconstruction error: {max_error}")
    print(f"Mean reconstruction error: {mean_error}")
    
    # 判断整体结果
    is_success = (max_error <= error_tol and u_ortho and v_ortho and b_form)
    
    if is_success:
        print("Test passed!")
    else:
        print("[ERROR] Test failed!")
        
    return is_success

if __name__ == '__main__':
    try:
        # 从args.txt读取矩阵维度
        with open('../args.txt', 'r') as file:
            args = file.read().split()
            M = int(args[0])
            N = int(args[1])
        
        res = verify_result(sys.argv[1], sys.argv[2], M, N)
        if not res:
            raise ValueError("[ERROR] Result verification failed")
    except Exception as e:
        print(e)
        sys.exit(1)