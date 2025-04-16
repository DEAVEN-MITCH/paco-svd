#!/usr/bin/python3
import sys
import numpy as np
import os # Import os

# 设置误差容限
relative_tol = 1e-3
absolute_tol = 1e-5
error_tol = 1e-3

def load_matrix(filename, shape):
    """加载二进制文件并重塑为指定形状的矩阵 (assumes row-major/C order)"""
    try:
        # Fortran unformatted stream files might have record markers (4-byte integers)
        # at the beginning and end of each record (write statement).
        # We need to account for these when reading.
        # Assuming one record per file for simplicity here.

        # Determine expected size
        dtype=np.float32
        expected_elements = np.prod(shape)
        expected_bytes = expected_elements * np.dtype(dtype).itemsize

        if not os.path.exists(filename):
             raise FileNotFoundError(f"File not found: {filename}")

        file_size = os.path.getsize(filename)
        # Files should now be exactly the expected size (row-major)
        if file_size != expected_bytes:
            print(f"[WARNING] File size mismatch for {os.path.basename(filename)}. Expected {expected_bytes} bytes (row-major), but got {file_size} bytes.")

        with open(filename, 'rb') as f:
            # Read the data
            data = np.fromfile(f, dtype=dtype, count=expected_elements)

            if data.size != expected_elements:
                 raise ValueError(f"Error loading {filename}: Expected {expected_elements} elements ({expected_bytes} bytes), but got {data.size} elements ({data.nbytes} bytes). File size is {file_size} bytes.")

             # Check for extra data
            remaining_data = f.read()
            if remaining_data:
                 print(f"[WARNING] Unexpected extra {len(remaining_data)} bytes found at the end of {os.path.basename(filename)} after reading expected data.")


        # Use default (row-major) order for reshape
        return data.reshape(shape)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filename}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e: # Catch other potential IOErrors
        print(f"[ERROR] Could not read file {filename}: {e}")
        sys.exit(1)


def verify_orthogonal(matrix, name):
    """验证矩阵是否正交"""
    # For Vt (N,N), it should be Vt @ Vt.T = I(N)
    # For U (M,M), it should be U @ U.T = I(M)
    rows, cols = matrix.shape
    if rows != cols:
        print(f"[WARNING] {name} is not a square matrix ({rows}x{cols}), cannot perform standard orthogonality check (M @ M.T = I). Skipping.")
        # Depending on context, you might want to check U[:,:K] @ U[:,:K].T = I(K) if M > N
        # or Vt[:K,:] @ Vt[:K,:].T = I(K) if N > M, but Fortran code generates square U/Vt.
        return False # Treat non-square as non-orthogonal for this check

    identity_dim = rows # Use rows for U (M,M) and Vt (N,N) as Fortran generates square matrices here
    I = np.eye(identity_dim, dtype=np.float32)

    try:
        product = matrix @ matrix.T # Check M @ M.T
    except Exception as e:
        print(f"[ERROR] Could not compute {name} @ {name}.T: {e}")
        return False

    is_orthogonal = np.allclose(product, I, rtol=relative_tol, atol=absolute_tol)
    if not is_orthogonal:
        print(f"[WARNING] {name} is not orthogonal ({name} @ {name}.T != I)")
        diff = np.abs(product - I)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        print(f"Max deviation from identity: {max_diff:.6e}, Avg deviation: {avg_diff:.6e}")
    else:
         print(f"[INFO] {name} orthogonality verified.")
    return is_orthogonal

def verify_bidiagonal(B, upper=True):
    """验证矩阵是否为（上）双对角矩阵"""
    M, N = B.shape
    is_bidiagonal = True
    max_off_diag_val = 0.0
    off_diag_indices = []

    for i in range(M):
        for j in range(N):
            is_main_diag = (i == j)
            is_super_diag = (j == i + 1) and upper
            is_sub_diag = (i == j + 1) and not upper # Corrected: sub-diagonal is i = j + 1

            # Check if the element is NOT on the allowed diagonals
            if not (is_main_diag or is_super_diag or is_sub_diag):
                if abs(B[i, j]) > absolute_tol:
                    if is_bidiagonal: # Only print the first warning message once
                         print(f"[WARNING] B is not strictly bidiagonal. Non-zero element found outside bidiagonal bands.")
                    is_bidiagonal = False
                    if abs(B[i,j]) > max_off_diag_val:
                        max_off_diag_val = abs(B[i,j])
                    if len(off_diag_indices) < 10: # Limit printing indices
                       off_diag_indices.append(((i, j), B[i, j]))


    if not is_bidiagonal:
        print(f"Max absolute value outside bidiagonal bands: {max_off_diag_val:.6e}")
        # Optionally print first few offending indices
        # print("First few non-zero off-band elements (index, value):")
        # for idx, val in off_diag_indices:
        #      print(f"  ({idx[0]},{idx[1]}): {val:.6e}")
    else:
        print("[INFO] B bidiagonal form verified (within absolute tolerance).")
    # This function now only checks the form, not comparing to golden B
    return is_bidiagonal

def verify_result(output_dir, golden_dir, M, N):
    """验证双对角化结果 (U, B, Vt) 的性质和重构精度"""
    print(f"Verifying results for M={M}, N={N}")
    print(f"Output directory (results being tested): {output_dir}")
    print(f"Golden directory (for original A): {golden_dir}")
    print(f"Tolerances: relative={relative_tol:.1e}, absolute={absolute_tol:.1e}, reconstruction_error={error_tol:.1e}")

    # --- Load Output Results (from your implementation) ---
    print("\nLoading implemented results...")
    u_path = os.path.join(output_dir, "U.bin")
    b_path = os.path.join(output_dir, "B.bin")
    vt_path = os.path.join(output_dir, "Vt.bin")
    try:
        U_out = load_matrix(u_path, (M, M))
        B_out = load_matrix(b_path, (M, N))
        Vt_out = load_matrix(vt_path, (N, N))
        print(f"Successfully loaded U, B, Vt from {output_dir}")
    except Exception as e:
        # Error already printed in load_matrix
        print(f"[ERROR] Failed to load one or more output files (U/B/Vt). Aborting verification.")
        return False # Abort if files can't be loaded

    # --- Load Original Matrix A (from where golden data was generated) ---
    # It's crucial this A matches the one used to generate the golden U, B, Vt
    a_gm_path = os.path.join(golden_dir, "../input/A_gm.bin") # Default expected path
    if not os.path.exists(a_gm_path):
       # Fallback if golden_dir is not the direct parent of 'input'
       # Assume 'input' is sibling to script's parent dir
       script_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
       a_gm_path_alt = os.path.join(script_parent_dir, "input/A_gm.bin")
       if os.path.exists(a_gm_path_alt):
           a_gm_path = a_gm_path_alt
           print(f"[INFO] Found A_gm.bin at alternative path: {a_gm_path}")
       else:
           print(f"[ERROR] Original matrix A_gm.bin not found.")
           print(f"  Tried: {os.path.abspath(a_gm_path)}")
           print(f"  Tried: {a_gm_path_alt}")
           print(f"  Ensure the 'input' directory with A_gm.bin exists relative to 'golden_dir' or the project root.")
           return False

    print(f"\nLoading original matrix A from {a_gm_path}...")
    try:
        A_original = load_matrix(a_gm_path, (M, N))
        print("Successfully loaded original matrix A.")
    except Exception as e:
        # Error already printed in load_matrix
        print(f"[ERROR] Failed to load original matrix A ({a_gm_path}). Aborting verification.")
        return False

    # --- Verification Steps ---
    print("\n--- Starting Verification ---")

    # 1. Verify Orthogonality of U_out and Vt_out
    print("\n1. Verifying Orthogonality...")
    u_ortho = verify_orthogonal(U_out, "U")
    vt_ortho = verify_orthogonal(Vt_out, "Vt") # Verify Vt directly

    # 2. Verify B_out is Bidiagonal (structurally)
    print("\n2. Verifying B is Bidiagonal...")
    # Assuming upper bidiagonal based on sgebrd standard output
    # Note: signs in B might differ from a different implementation, but structure should hold
    b_form = verify_bidiagonal(B_out, upper=True)

    # 3. Verify Reconstruction Accuracy (A_original vs U_out @ B_out @ Vt_out)
    print("\n3. Verifying Reconstruction A = U @ B @ Vt...")
    reconstruction_passed = False
    max_error = float('inf')
    mean_error = float('inf')
    try:
        A_reconstructed = U_out @ B_out @ Vt_out
        diff = np.abs(A_reconstructed - A_original)
        max_error = np.max(diff)
        mean_error = np.mean(diff)
        reconstruction_passed = max_error <= error_tol

        print(f"Maximum reconstruction error (|A_rec - A_orig|): {max_error:.6e}")
        print(f"Mean reconstruction error: {mean_error:.6e}")
        if reconstruction_passed:
            print(f"[INFO] Reconstruction error is within tolerance ({error_tol:.4e}).")
        else:
            print(f"[WARNING] Reconstruction error ({max_error:.6e}) EXCEEDS tolerance ({error_tol:.4e}).")
    except Exception as e:
         print(f"[ERROR] Could not compute or compare reconstructed A: {e}")
         print(f"Shapes: U={U_out.shape}, B={B_out.shape}, Vt={Vt_out.shape}, A_orig={A_original.shape}")


    # --- Final Verdict ---
    print("\n--- Verification Summary ---")
    print(f"1. U Orthogonality Check:      {'PASSED' if u_ortho else 'FAILED'}")
    print(f"2. Vt Orthogonality Check:     {'PASSED' if vt_ortho else 'FAILED'}")
    print(f"3. B Bidiagonal Form Check:    {'PASSED' if b_form else 'FAILED'}")
    print(f"4. Reconstruction Accuracy Check: {'PASSED' if reconstruction_passed else 'FAILED'}")

    is_success = u_ortho and vt_ortho and b_form and reconstruction_passed

    print("\n--- Overall Result ---")
    if is_success:
        print("✅ Test PASSED!")
    else:
        print("❌ Test FAILED!")

    return is_success

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <output_directory> <golden_directory>")
        print("  <output_directory>: Directory containing U.bin, B.bin, Vt.bin to test.")
        print("  <golden_directory>: Directory containing U_golden.bin, etc., and parent of 'input/A_gm.bin'.")
        sys.exit(1)

    output_dir_arg = sys.argv[1]
    golden_dir_arg = sys.argv[2]

    # Determine args.txt path relative to this script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args_file_path = os.path.join(script_dir, '../args.txt') # Assume in parent dir

    M_arg, N_arg = -1, -1 # Initialize

    try:
        # Read matrix dimensions from args.txt
        print(f"Reading matrix dimensions from: {args_file_path}")
        if not os.path.exists(args_file_path):
             raise FileNotFoundError(f"args.txt not found at expected location: {args_file_path}")

        with open(args_file_path, 'r') as file:
            line = file.readline().strip()
            if not line:
                 raise ValueError("args.txt is empty or contains only whitespace.")
            args = line.split()
            if len(args) < 2:
                 raise ValueError(f"args.txt needs two numbers (M N), found: '{line}'")
            try:
                M_arg = int(args[0])
                N_arg = int(args[1])
                if M_arg <= 0 or N_arg <= 0:
                    raise ValueError("M and N must be positive integers.")
            except ValueError:
                 raise ValueError(f"Could not parse M and N as integers from args.txt line: '{line}'")


        print(f"Read M={M_arg}, N={N_arg}")

        # Validate input directories exist
        if not os.path.isdir(output_dir_arg):
            print(f"[ERROR] Output directory does not exist: {output_dir_arg}")
            sys.exit(1)
        if not os.path.isdir(golden_dir_arg):
             print(f"[ERROR] Golden directory does not exist: {golden_dir_arg}")
             sys.exit(1)


        # Run verification
        verification_passed = verify_result(output_dir_arg, golden_dir_arg, M_arg, N_arg)

        # Exit with appropriate status code
        sys.exit(0 if verification_passed else 1)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during verification:")
        import traceback
        traceback.print_exc()
        sys.exit(1)