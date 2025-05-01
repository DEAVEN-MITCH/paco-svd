#include <cstdint>
#include "tiling/platform/platform_ascendc.h"
#include <tiling/tiling_api.h>

namespace
{
    struct SVDSubmatrixInfo
    {
        // the submatrix is [start_col:end_col-1,start_col:end_col] if end_col !=LDN else [start_col:end_col,start_col:end_col]
        uint16_t start_col, end_col;
    };

    // struct SVDStack {
    // SVDSubmatrixInfo* stackPtr;
    // uint16_t stackSize;
    // };

    struct SVDTiling
    {
        // TCubeTiling tiling;
        uint16_t stackSize;
        uint32_t offset;//Bytes
    };
    constexpr int getFirstStackElementOffset(int svdBlockNum=1){
        return (sizeof(SVDTiling) + 31) / 32 * 32 * svdBlockNum;
    }
}
int getTilingSize(int N, int svdBlockNum = 1)
{
    // currently one aiv,one aic
    // compute the basic matrix quantity
    auto totalStackSize = N / 3 + 1;
    // auto aivNum=svdBlockNum;
    // auto formerStackSize=(totalStackSize+aivNum-1)/aivNum;
    // auto tailStackSize=totalStackSize/aivNum;
    // auto formerStackNum=totalStackSize%aivNum;
    // auto tailStackNum=aivNum-formerStackNum;
    // formerStack has stackSize of formerStackSize.

    return getFirstStackElementOffset(svdBlockNum) + sizeof(SVDSubmatrixInfo) * totalStackSize;
}

// Tiling: SVDTiling+SVDStack,currently one aiv ,one aic
void GenerateTiling(int N, int svdBlockNum, uint8_t *TilingHost)
{
    SVDTiling *tiling = reinterpret_cast<SVDTiling *>(TilingHost);
    auto totolStackSize = N / 3 + 1;
    tiling->stackSize = totolStackSize;
    tiling->offset = getFirstStackElementOffset(svdBlockNum);
    SVDSubmatrixInfo *svdStack = reinterpret_cast<SVDSubmatrixInfo *>(TilingHost + getFirstStackElementOffset(svdBlockNum));
    auto remainder = N % 3;
    // remainder==0,N/3-1个2x3+1x2+1x1
    // remainder==1,N/3个2x3+1x1
    // remainder==2,N/3个2x3+2x2
    for (int i = 0; i < totolStackSize - 2; i++)
    {
        svdStack[i].start_col = i * 3;
        svdStack[i].end_col = i * 3 + 3;
    }
    // init stack.stackSize-2 th SVDSubmatrixInfo
    int i = totolStackSize - 2;
    if (remainder == 0)
    {
        svdStack[i].start_col = i * 3;
        svdStack[i].end_col = i * 3 + 2;
    }
    else
    {
        svdStack[i].start_col = i * 3;
        svdStack[i].end_col = i * 3 + 3;
    }
    // init stack.stackSize-1 th SVDSubmatrixInfo
    i = totolStackSize - 1;
    switch (remainder)
    {
    case 0:
        svdStack[i].start_col = i * 3 - 1;
        svdStack[i].end_col = i * 3;
        break;
    case 1:
        svdStack[i].start_col = i * 3;
        svdStack[i].end_col = i * 3 + 1;
        break;
    case 2:
        svdStack[i].start_col = i * 3;
        svdStack[i].end_col = i * 3 + 2;
        break;
    }
    return;
}
