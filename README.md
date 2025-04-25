# run
./run.sh to build in debug mode
./run.sh r to clean and rebuild in debug mode
./run.sh Re to clean and rebuild in release mode
./run.sh to generate new input data according to args.txt
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
- under development

