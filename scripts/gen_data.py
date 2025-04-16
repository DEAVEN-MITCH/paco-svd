import subprocess
import os

def write_args_file(m, n, path="../args.txt"):
    with open(path, "w") as f:
        f.write(f"{m} {n}\n")

def compile_fortran(source_file, exe_name="bidiag_gen", output_dir="../test_bin"):
    os.makedirs(output_dir, exist_ok=True)
    exe_path = os.path.join(output_dir, exe_name)

    compile_cmd = [
        "gfortran",
        source_file,
        "-o", exe_path,
        "-llapack", "-lblas"
    ]
    print("Compiling Fortran source...")
    subprocess.run(compile_cmd, check=True)
    return exe_path

def run_executable(exe_path):
    print("Running executable...")
    subprocess.run([exe_path], check=True)

if __name__ == "__main__":
    # M, N = 5, 5  # 你可以根据需求设置
    # write_args_file(M, N)

    exe_path = compile_fortran("../scripts/bidiag_gen.f90")
    run_executable(exe_path)
