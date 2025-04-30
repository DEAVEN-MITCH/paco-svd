#include <tiling/tiling_api.h>
#include "tiling/platform/platform_ascendc.h"



size_t getTilingSize(int N, int svdBlockNum=1);
void GenerateTiling(int N, int blockDim, uint8_t *TilingHost);
struct SVDSubmatrixInfo{
    //the submatrix is [start_col:end_col-1,start_col:end_col] if end_col !=LDN else [start_col:end_col,start_col:end_col]
    uint16_t start_col,end_col;
};
struct SVDStack{
    SVDSubmatrixInfo* stackPtr;
    uint16_t stackSize;
}

struct SVDTiling{
    SVDStack stack;
    matmul_tiling::MatmulApiTiling matmulTiling;
};
