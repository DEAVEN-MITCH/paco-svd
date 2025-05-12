# run
./run.sh to build in debug mode
./run.sh r to clean and rebuild in debug mode
./run.sh Re to clean and rebuild in release mode
./run.sh v to generate new input data according to args.txt
./run.sh t to run test script
./run.sh main run the svd_main of debug version
./myprof.sh d to run profiler of debug version
./myprof.sh r to run profiler of release version

# args.txt
M N
of matrix size

# upper bidiagonalization
- using Golub-Kahan methods
- tested
- parallel
- small matrix performs better when the implementation is not parallel
- setting BUFFER_NUM to 2 result no better performance than 1,but smaller BufferSIze
- bigger BufferSize without intra-core tiling performs better than smaller BufferSize with intra-core tiling
- LHC and 3 step bidiagonalization requires extra global memory,may be implemented later
- using aiv to implement householder transformation.If we want to use aic to apply householder transformation,we need extra global memory and aic-aiv sync ,may be implemented later

# BDC
- using high-level API matmul 
- not using QR subroutine to solve leaf subproblems
- not parallel yet,but it's easy to implement based on the existing code
- not tiling yet,testing is insufficient
- Secular equation root finder is not enough feasible yet


