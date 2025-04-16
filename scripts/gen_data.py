import subprocess
import os
import numpy as np # Import numpy

def write_args_file(m, n, path="../args.txt"):
    print(f"Writing M={m}, N={n} to {path}")
    with open(path, "w") as f:
        f.write(f"{m} {n}\n")

def compile_fortran(source_file, exe_name="bidiag_gen", output_dir="../test_bin"):
    os.makedirs(output_dir, exist_ok=True)
    exe_path = os.path.join(output_dir, exe_name)
    source_path = os.path.abspath(source_file) # Use absolute path for source

    # Determine compiler based on availability
    compiler = None
    try:
        subprocess.run(["gfortran", "--version"], check=True, capture_output=True)
        compiler = "gfortran"
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            subprocess.run(["ifort", "--version"], check=True, capture_output=True)
            compiler = "ifort"
        except (FileNotFoundError, subprocess.CalledProcessError):
             print("[ERROR] No suitable Fortran compiler (gfortran or ifort) found in PATH.")
             return None

    print(f"Using Fortran compiler: {compiler}")

    compile_cmd = [
        compiler,
        source_path,
        "-o", exe_path,
        "-llapack", "-lblas" # Standard LAPACK/BLAS linking
    ]
    # Add OpenMP flag if using gfortran (optional, but common)
    # if compiler == "gfortran":
    #     compile_cmd.insert(1, "-fopenmp")
    # elif compiler == "ifort":
    #      compile_cmd.insert(1, "-qopenmp") # Intel OpenMP flag

    print(f"Compiling Fortran source: {' '.join(compile_cmd)}")
    try:
        result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        print("Compilation successful.")
        if result.stdout: print("Compiler output:", result.stdout)
        if result.stderr: print("Compiler warnings/errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("[ERROR] Fortran compilation failed.")
        print("Command:", e.cmd)
        print("Return code:", e.returncode)
        print("Output:", e.stdout)
        print("Error:", e.stderr)
        return None
    except FileNotFoundError:
         print(f"[ERROR] Compiler '{compiler}' not found.")
         return None

    return exe_path

def run_executable(exe_path):
    if not exe_path or not os.path.exists(exe_path):
         print("[ERROR] Executable path is invalid or file does not exist.")
         return False
    print(f"Running executable: {exe_path}")
    try:
        # Run in the directory containing the executable for relative paths in Fortran code to work
        exe_dir = os.path.dirname(exe_path)
        result = subprocess.run([exe_path], check=True, capture_output=True, text=True, cwd=exe_dir)
        print("Executable finished successfully.")
        if result.stdout: print("Executable output:", result.stdout)
        if result.stderr: print("Executable errors/warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print("[ERROR] Executable failed.")
        print("Command:", e.cmd)
        print("Return code:", e.returncode)
        print("Output:", e.stdout)
        print("Error:", e.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while running the executable: {e}")
        return False

def convert_fortran_bin_to_row_major(filepath, shape, dtype=np.float32):
    """Reads a Fortran (column-major) binary file and overwrites it in row-major format."""
    print(f"  Converting {filepath} (Shape: {shape})... ", end="")
    try:
        # 1. Read data using Fortran order
        with open(filepath, 'rb') as f:
            data_flat = np.fromfile(f, dtype=dtype)

        expected_elements = np.prod(shape)
        if data_flat.size != expected_elements:
             # Check for potential record markers (simple heuristic)
             expected_bytes = expected_elements * np.dtype(dtype).itemsize
             if data_flat.size * np.dtype(dtype).itemsize == expected_bytes + 8:
                 # Assume markers exist, try reading without them
                 print(f"[Warning] Possible Fortran record markers detected. Attempting read without markers.")
                 with open(filepath, 'rb') as f:
                     f.seek(4) # Skip start marker
                     data_flat = np.fromfile(f, dtype=dtype, count=expected_elements)
                 if data_flat.size != expected_elements:
                      raise ValueError(f"Read failed even after attempting to skip markers. Expected {expected_elements}, got {data_flat.size}.")
             else:
                 raise ValueError(f"Incorrect number of elements read. Expected {expected_elements}, got {data_flat.size}. File size might be wrong.")

        # 2. Reshape using Fortran order to get correct structure in memory
        matrix = data_flat.reshape(shape, order='F')

        # 3. Write back using default (C/row-major) order
        with open(filepath, 'wb') as f:
            matrix.tofile(f) # np.tofile uses C order by default

        print("Done.")
        return True
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {filepath}")
        return False
    except ValueError as e:
        print(f"\n[ERROR] Error processing {filepath}: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error converting {filepath}: {e}")
        return False


if __name__ == "__main__":
    # --- Configuration ---
    M, N = 1024, 1024 # Example dimensions
    args_file = "../args.txt"
    fortran_source = "../scripts/bidiag_gen.f90"
    input_dir = "../input"
    output_dir = "../output" # Where Fortran writes U, B, Vt
    exe_dir = "../test_bin"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(exe_dir, exist_ok=True)

    # --- Generate args.txt ---
    write_args_file(M, N, args_file)

    # --- Compile Fortran ---
    exe_path = compile_fortran(fortran_source, output_dir=exe_dir)

    if not exe_path:
        print("Exiting due to compilation failure.")
        exit(1)

    # --- Run Fortran Executable ---
    success = run_executable(exe_path)

    if not success:
        print("Exiting due to executable failure.")
        exit(1)

    # --- Convert Output Files to Row-Major Order ---
    print("\nConverting Fortran output files to row-major format...")

    a_path = os.path.join(input_dir, "A_gm.bin")
    u_path = os.path.join(output_dir, "U_golden.bin")
    b_path = os.path.join(output_dir, "B_golden.bin")
    vt_path = os.path.join(output_dir, "Vt_golden.bin")

    a_shape = (M, N)
    u_shape = (M, M)
    b_shape = (M, N)
    vt_shape = (N, N)

    all_converted = True
    if not convert_fortran_bin_to_row_major(a_path, a_shape): all_converted = False
    if not convert_fortran_bin_to_row_major(u_path, u_shape): all_converted = False
    if not convert_fortran_bin_to_row_major(b_path, b_shape): all_converted = False
    if not convert_fortran_bin_to_row_major(vt_path, vt_shape): all_converted = False

    if all_converted:
        print("All files converted successfully.")
    else:
        print("[ERROR] One or more files failed conversion. Check errors above.")
        exit(1)

    print("\nData generation and conversion complete.")
