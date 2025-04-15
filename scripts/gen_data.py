import numpy as np
from scipy.linalg import bidiagonal

def gen_golden_data_bidiag():
    # 读取参数
    with open('../args.txt', 'r') as file:
        args = file.read().split()
        M = int(args[0])
        N = int(args[1])

    # 生成随机输入矩阵 A
    A = np.random.uniform(-10, 10, [M, N]).astype(np.float32)
    
    # 计算双对角化分解
    # scipy.linalg.bidiagonal 返回 (U, B, Vh)
    U, B, Vt = bidiagonal(A, upper=True, overwrite_a=False)
    
    # 转换为 float32 类型
    U = U.astype(np.float32)
    B = B.astype(np.float32)
    Vt = Vt.astype(np.float32)
    
    # 保存输入矩阵
    A.tofile("../input/A_gm.bin")
    
    # 保存golden结果
    U.tofile("../output/U_golden.bin")
    B.tofile("../output/B_golden.bin")
    Vt.tofile("../output/Vt_golden.bin")
    
    # 保存验证用的重构矩阵
    A_reconstructed = U @ B @ Vt
    A_reconstructed.tofile("../output/A_reconstructed_golden.bin")

if __name__ == "__main__":
    gen_golden_data_bidiag()