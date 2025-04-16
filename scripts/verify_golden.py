#!/usr/bin/python3
import sys
import numpy as np
import os

# --- Configuration ---
# Tolerances for comparing floating-point numbers
absolute_tol = 1e-5 # Tolerance for checking if off-diagonal elements are zero
error_tol = 1e-3 # Maximum allowed absolute error for reconstruction

# Relative paths (adjust if your structure is different)
ARGS_FILE = '../args.txt'
INPUT_DIR = '../input'
OUTPUT_DIR = '../output' # Directory where bidiag_gen.f90 writes its output

# --- Helper Function (copied from verify_result.py for robustness) ---
def load_matrix(filename, shape, dtype=np.float32):
    """Loads a matrix from a binary file (assumes row-major/C order)."""
    try:
        expected_elements = np.prod(shape)
        expected_bytes = expected_elements * np.dtype(dtype).itemsize

        if not os.path.exists(filename):
             raise FileNotFoundError(f"File not found: {filename}")

        file_size = os.path.getsize(filename)

        # Files should now be exactly the expected size (row-major, no markers)
        if file_size != expected_bytes:
            print(f"[WARNING] File size mismatch for {os.path.basename(filename)}. Expected {expected_bytes} bytes (row-major), but got {file_size} bytes.")
            # Consider raising an error or handling potential issues

        with open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=dtype, count=expected_elements)

            if data.size != expected_elements:
                 raise ValueError(f"Error loading {filename}: Expected {expected_elements} elements, but read {data.size}. File size is {file_size} bytes.")

            # Check for extra data
            remaining_data = f.read()
            if remaining_data:
                print(f"[WARNING] Unexpected extra {len(remaining_data)} bytes found at the end of {os.path.basename(filename)} after reading expected data.")

        # Reshape using default C (row-major) order!
        return data.reshape(shape) # Removed order='F'

    except FileNotFoundError:
        print(f"[ERROR] File not found: {filename}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not read or process file {filename}: {e}")
        sys.exit(1)

# --- Function to Inspect B ---
def inspect_b_matrix(B, name="B", tol=1e-5):
    """Prints key information about the B matrix."""
    m, n = B.shape
    k = min(m, n)
    print(f"\n--- Inspecting Matrix: {name} ({B.dtype}) ---")
    print(f"Shape: {B.shape}")
    print(f"Max value: {np.max(B):.6e}")
    print(f"Min value: {np.min(B):.6e}")
    print(f"Mean value: {np.mean(B):.6e}")

    # Extract diagonals
    diag = np.diag(B)
    super_diag = np.diag(B, k=1) # k=1 for the super-diagonal

    print(f"\nMain diagonal (D) - first 10 elements:")
    print(diag[:10])
    if k > 10:
        print(f"Main diagonal (D) - last 10 elements:")
        print(diag[-10:])

    # Super-diagonal exists only if K > 1
    if k > 1:
      print(f"\nSuper-diagonal (E) - first 10 elements:")
      print(super_diag[:10])
      if k -1 > 10: # k-1 is the length of super_diag
          print(f"Super-diagonal (E) - last 10 elements:")
          print(super_diag[-10:])
    else:
        print("\nSuper-diagonal (E): Not applicable for this shape.")

    # Check for significant off-band elements
    off_band_indices = []
    max_off_band_val = 0.0
    for i in range(m):
        for j in range(n):
            # Check if element is NOT on main diagonal or super-diagonal
            if i != j and j != i + 1:
                if abs(B[i, j]) > tol:
                    off_band_indices.append(((i, j), B[i, j]))
                    if abs(B[i,j]) > max_off_band_val:
                        max_off_band_val = abs(B[i,j])

    if not off_band_indices:
        print(f"\n[INFO] All elements outside the main and super-diagonal are within tolerance ({tol:.1e}).")
    else:
        print(f"\n[WARNING] Found {len(off_band_indices)} elements outside main/super-diagonal with absolute value > {tol:.1e}.")
        print(f"  Maximum absolute value found off-band: {max_off_band_val:.6e}")
        print("  First 10 off-band elements (index, value):")
        for k, (idx, val) in enumerate(off_band_indices[:10]):
            print(f"    ({idx[0]}, {idx[1]}): {val:.6e}")
    print("--- End Inspection ---")


# --- Main Verification Logic ---
def verify_golden_reconstruction(m, n):
    """Loads golden files and verifies A = U * B * Vt."""
    # Set print options for numpy arrays
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    print(f"Verifying golden reconstruction for M={m}, N={n}")
    print(f"Loading files from INPUT_DIR='{INPUT_DIR}' and OUTPUT_DIR='{OUTPUT_DIR}'")

    # Define file paths
    a_path = os.path.join(INPUT_DIR, "A_gm.bin")
    u_path = os.path.join(OUTPUT_DIR, "U_golden.bin")
    b_path = os.path.join(OUTPUT_DIR, "B_golden.bin")
    vt_path = os.path.join(OUTPUT_DIR, "Vt_golden.bin")

    # Load matrices
    print("\nLoading matrices...")
    try:
        A_original = load_matrix(a_path, (m, n))
        print(f"  Loaded A_original ({A_original.shape}) from {a_path}")
        U_golden = load_matrix(u_path, (m, m))
        print(f"  Loaded U_golden ({U_golden.shape}) from {u_path}")
        B_golden = load_matrix(b_path, (m, n))
        print(f"  Loaded B_golden ({B_golden.shape}) from {b_path}")
        Vt_golden = load_matrix(vt_path, (n, n))
        print(f"  Loaded Vt_golden ({Vt_golden.shape}) from {vt_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to load one or more required files. Aborting.")
        return False # Abort if loading fails

    # +++ Inspect B_golden +++
    inspect_b_matrix(B_golden, name="B_golden", tol=absolute_tol)
    # ++++++++++++++++++++++++

    # Perform reconstruction
    print("\nCalculating reconstruction: A_rec = U_golden @ B_golden @ Vt_golden...")
    try:
        A_reconstructed = U_golden @ B_golden @ Vt_golden
        print("  Calculation complete.")
    except Exception as e:
        print(f"\n[ERROR] Matrix multiplication failed: {e}")
        print(f"  Shapes: U={U_golden.shape}, B={B_golden.shape}, Vt={Vt_golden.shape}")
        return False

    # Compare A_original and A_reconstructed
    print("\nComparing A_original and A_reconstructed...")
    diff = np.abs(A_reconstructed - A_original)
    max_error = np.max(diff)
    mean_error = np.mean(diff)

    print(f"  Maximum absolute difference: {max_error:.6e}")
    print(f"  Mean absolute difference:  {mean_error:.6e}")

    # Check against tolerance
    if max_error <= error_tol:
        print(f"\n[INFO] Reconstruction error ({max_error:.6e}) is within tolerance ({error_tol:.4e}).")
        print("✅ Verification PASSED!")
        return True
    else:
        print(f"\n[WARNING] Reconstruction error ({max_error:.6e}) EXCEEDS tolerance ({error_tol:.4e}).")
        print("❌ Verification FAILED!")
        return False

if __name__ == "__main__":
    print("--- Golden Data Reconstruction Verification ---")

    # Find args.txt relative to the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args_file_path = os.path.join(script_dir, ARGS_FILE)

    M, N = -1, -1 # Initialize

    # Read dimensions M and N from args.txt
    try:
        print(f"\nReading matrix dimensions from: {args_file_path}")
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
                M = int(args[0])
                N = int(args[1])
                if M <= 0 or N <= 0:
                    raise ValueError("M and N must be positive integers.")
            except ValueError:
                 raise ValueError(f"Could not parse M and N as integers from args.txt line: '{line}'")

        print(f"Read M={M}, N={N}")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to read or parse {args_file_path}: {e}")
        sys.exit(1)


    # Run the verification
    try:
        passed = verify_golden_reconstruction(M, N)
        sys.exit(0 if passed else 1) # Exit with 0 on success, 1 on failure

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during verification:")
        import traceback
        traceback.print_exc()
        sys.exit(1) 