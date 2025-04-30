#include "svd_tiling.h"

size_t getTilingSize(int N, int svdBlockNum)
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

    return (sizeof(SVDTiling) + sizeof(SVDSubmatrixInfo) * totalStackSize);
}

// Tiling: SVDTiling+SVDStack,currently one aiv ,one aic
void GenerateTiling(int N, int blockDim, uint8_t *TilingHost)
{
    SVDTiling *tiling = static_cast<SVDTiling *>(TilingHost);
    tiling->stack.stackPtr = TilingHost + sizeof(SVDTiling);
    tiling->stack.stackSize = N / 3 + 1;
    auto &svdStack = tiling->stack;
    auto remainder = N % 3;
    // remainder==0,N/3-1个2x3+1x2+1x1
    // remainder==1,N/3个2x3+1x1
    // remainder==2,N/3个2x3+2x2
    for (int i = 0; i < tiling->stack.stackSize - 2; i++)
    {
        svdStack.stackPtr[i].start_col = i * 3;
        svdStack.stackPtr[i].end_col = i * 3 + 3;
    }
    // init stack.stackSize-2 th SVDSubmatrixInfo
    int i = tiling->stack.stackSize - 2;
    if (remainder == 0)
    {
        svdStack.stackPtr[i].start_col = i * 3;
        svdStack.stackPtr[i].end_col = i * 3 + 2;
    }
    else
    {
        svdStack.stackPtr[i].start_col = i * 3;
        svdStack.stackPtr[i].end_col = i * 3 + 3;
    }
    // init stack.stackSize-1 th SVDSubmatrixInfo
    i = tiling->stack.stackSize - 1;
    switch (remainder)
    {
    case 0:
        svdStack.stackPtr[i].start_col = i * 3 - 1;
        svdStack.stackPtr[i].end_col = i * 3;
        break;
    case 1:
        svdStack.stackPtr[i].start_col = i * 3;
        svdStack.stackPtr[i].end_col = i * 3 + 1;
        break;
    case 2:
        svdStack.stackPtr[i].start_col = i * 3;
        svdStack.stackPtr[i].end_col = i * 3 + 2;
        break;
    }
    return;
}
