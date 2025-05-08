#include "kernel_operator.h"
#include <lib/matmul_intf.h>
#include <limits>
#define NotParallelQuiter      \
    if constexpr (!ifParallel) \
    {                          \
        if (blockIdx != 0)     \
        {                      \
            return;            \
        }                      \
    }
#define singleDumpTensor(tensor, tensorNum)      \
    if (blockIdx == 0)                           \
    {                                            \
        printf("dump" #tensor "\n");             \
        DumpTensor(tensor, __LINE__, tensorNum); \
    }

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr float EPS = std::numeric_limits<float>::epsilon();
constexpr uint32_t sizeOfFloat = sizeof(float);
constexpr uint32_t sizeOfUint32_t = sizeof(uint32_t);
constexpr int32_t BlockSize = 32;
constexpr int32_t BlockFloatCnt = BlockSize / sizeOfFloat;
constexpr int32_t SizePerOperation = 256;
constexpr int32_t BlockNumPerOperation = SizePerOperation / BlockSize;
constexpr int32_t MaxIter = 10;
using namespace AscendC;
using namespace matmul;

namespace
{
    __aicore__ inline float sign(float a, float b)
    {
        return b < 0 ? -a : a;
    }
    __aicore__ inline void swap(float &a, float &b)
    {
        float tmp = a;
        a = b;
        b = tmp;
    }
    __aicore__ inline float fabs(float a)
    {
        return a < 0 ? -a : a;
    }
    // __aicore__ inline float max(float a, float b)
    // {
    //     return a >= b ? a : b;
    // }
}
namespace
{
    struct SVDTiling
    {
        TCubeTiling matmultiling;
        uint16_t stackSize;
        uint32_t offset; // Bytes
    };
    struct SVDSubmatrixInfo
    {
        // the submatrix is [start_col:end_col-1,start_col:end_col] if end_col !=LDN else [start_col:end_col,start_col:end_col]
        uint16_t start_col, end_col;
    };
    // currently one aiv,one aic
    __aicore__ inline void CopyTiling(SVDTiling *tiling, GM_ADDR *svdStack, GM_ADDR tilingGM, int thread_id = 0)
    {
        uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
        auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

        for (uint32_t i = 0; i < sizeof(SVDTiling) / sizeOfUint32_t; i++, ptr++)
        {
            *ptr = *(tiling32 + i);
        }
        *svdStack = tilingGM + tiling->offset;
        return;
    }
    using BDCMatmulType = Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                                 MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, true>,
                                 MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>>;
}
template <bool ifVecTiling = false, bool ifParallel = false>
class BDC
{
public:
    __aicore__ inline BDC() : blockIdx(AscendC::GetBlockIdx()), blockNum(AscendC::GetBlockNum()) {}
    __aicore__ inline void init(int M, int N, GM_ADDR a, GM_ADDR u, GM_ADDR vt, GM_ADDR d, GM_ADDR e, GM_ADDR qt, GM_ADDR wt, GM_ADDR idx, GM_ADDR svdStack, GM_ADDR workspace, SVDTiling *tiling, TPipe &pipe, BDCMatmulType &inputmm)
    {
        NotParallelQuiter;
        ASSERT(M >= N && "M must be greater than or equal to N");
        LDM = M;
        LDN = N;
        svdTiling = tiling;
        mm = &inputmm;
        svdStackGm.SetGlobalBuffer((__gm__ uint16_t *)svdStack, tiling->stackSize);
        tmpGm.SetGlobalBuffer((__gm__ float *)a, M * N);
        uGm.SetGlobalBuffer((__gm__ float *)u, M * M);
        vtGm.SetGlobalBuffer((__gm__ float *)vt, N * N);
        dGm.SetGlobalBuffer((__gm__ float *)d, N);
        eGm.SetGlobalBuffer((__gm__ float *)e, N - 1);
        qtGm.SetGlobalBuffer((__gm__ float *)qt, N * N);
        wtGm.SetGlobalBuffer((__gm__ float *)wt, N * N);
        stGm.SetGlobalBuffer((__gm__ float *)(qt + sizeOfFloat * N * N), N);
        gtGm.SetGlobalBuffer((__gm__ float *)(wt + sizeOfFloat * N * N), N);
        idxqGm.SetGlobalBuffer((__gm__ uint32_t *)idx, N);
        idx += N * sizeOfFloat;
        fGm.SetGlobalBuffer((__gm__ float *)idx, N);
        idx += N * sizeOfFloat;
        lGm.SetGlobalBuffer((__gm__ float *)idx, N);
        idx += N * sizeOfFloat;
        sigmaGm.SetGlobalBuffer((__gm__ float *)idx, N);
        idx += N * sizeOfFloat;
        idxpGm.SetGlobalBuffer((__gm__ uint32_t *)idx, N);
        idx += N * sizeOfFloat;
        idxGm.SetGlobalBuffer((__gm__ uint32_t *)idx, N);
        idx += N * sizeOfFloat;
        idxcGm.SetGlobalBuffer((__gm__ uint32_t *)idx, N);
        idx += N * sizeOfFloat;
        coltypGm.SetGlobalBuffer((__gm__ uint32_t *)idx, N);
        idx += N * sizeOfFloat;
        zGm.SetGlobalBuffer((__gm__ float *)idx, N);
        idx += N * sizeOfFloat;
        dsigmaGm.SetGlobalBuffer((__gm__ float *)idx, N);
        pipe.InitBuffer(copyBind, BUFFER_NUM, N * sizeOfFloat + 32);
        pipe.InitBuffer(inQueue, BUFFER_NUM, N * sizeOfFloat + 32);
        pipe.InitBuffer(outQueue, BUFFER_NUM, N * sizeOfFloat + 32);
        pipe.InitBuffer(tmpBuf1, 8 * N + 96); // for sort
        pipe.InitBuffer(tmpBuf2, 8 * N + 96); // for sort

        // pipe.InitBuffer(workspaceBuf, blockNum * 32);

        // workspace = workspaceBuf.Get<int32_t>();
    }
    __aicore__ inline void Process()
    {
        NotParallelQuiter;
        // reduction requires rotation in the end,may not be worth it
        // simply no reduction ,as sbdsdc does
        initQWt();
        if (LDN == 2)
        {
            if (blockIdx == 0)
            {
                compute_2x2_svd(qtGm, wtGm, dGm, eGm, idxqGm, fGm, lGm);
            }
            updateUVt();

            return;
        }
        else if (LDN == 1)
        {
            if (blockIdx == 0)
            {
                compute_1x1_svd(qtGm, wtGm, dGm, idxqGm, fGm, lGm);
            }
            updateUVt();

            return;
        }
        printf("not terminate early,LDN:%d,g_coreType==AIV:%d\n,blockIdx:%d\n", LDN, g_coreType == AIV, blockIdx);
        uint16_t stackSize = svdTiling->stackSize;
        // init leaf node
        for (auto i = 0; i < stackSize; i++)
        {
            compute_base_case_svd(getSVDSubmatrixInfo(i));
        }
        // merge
        while (stackSize != 1)
        {
            uint16_t stack_idx = 0, newStackSize = 0;
            while (stack_idx < stackSize - 1)
            {
                // have more than two subMatrix to merge
                const SVDSubmatrixInfo leftSubMatrix = getSVDSubmatrixInfo(stack_idx++);
                const SVDSubmatrixInfo rightSubMatrix = getSVDSubmatrixInfo(stack_idx++);
                SVDSubmatrixInfo mergedSubMatrix;
                mergedSubMatrix.start_col = leftSubMatrix.start_col;
                mergedSubMatrix.end_col = rightSubMatrix.end_col;
                MergeSubMatrix(leftSubMatrix, rightSubMatrix);
                setSVDSubmatrixInfo(newStackSize++, mergedSubMatrix);
            }
            if (stack_idx == stackSize - 1)
            // last tailing submatrix not merged
            {
                setSVDSubmatrixInfo(newStackSize++, getSVDSubmatrixInfo(stack_idx));
            }
            // update the condition
            stackSize = newStackSize;
        }
        if constexpr (!ifParallel)
        {
            rearrange_qwtAccordingToIdxq();
            updateUVt();
        }
    }

private:
    __aicore__ inline void initQWt()
    {
        // use aiv's scalars only
        //  初始化q和wt为单位矩阵
        if constexpr (!ifParallel)
        {

            if (blockIdx != 0)
                return;
            for (auto i = 0; i < LDN; i++)
            {
                qtGm(i * LDN + i) = 1.0f;
                wtGm(i * LDN + i) = 1.0f;
            }
        }

        else
        {

            for (int32_t i = blockIdx; i < LDN; i += blockNum)
            {
                qtGm(i * LDN + i) = 1.0f;
                wtGm(i * LDN + i) = 1.0f;
            }
        }
    }

    __aicore__ inline void MergeSubMatrix(const SVDSubmatrixInfo &leftSubMatrix, const SVDSubmatrixInfo &rightSubMatrix)
    {
        printf("[MergeSubMatrix] left(%d,%d), right(%d,%d)\n", leftSubMatrix.start_col, leftSubMatrix.end_col, rightSubMatrix.start_col, rightSubMatrix.end_col);
        // acquire singular matrix of subproblems
        bool isSquare = rightSubMatrix.end_col == LDN;
        auto leftColNum = leftSubMatrix.end_col - leftSubMatrix.start_col;    // nl cols,nl-1 rows
        auto rightColNum = rightSubMatrix.end_col - rightSubMatrix.start_col; // nr cols,nr-1+isSquare rows
        auto totalColNum = leftColNum + rightColNum;                          // total n cols,n-1+isSquare rows;
        auto alpha = dGm(leftSubMatrix.end_col - 1);
        auto beta = eGm(leftSubMatrix.end_col - 1);
        uint16_t k;
        auto d = dGm[leftSubMatrix.start_col];
        auto z = zGm[leftSubMatrix.start_col];
        GlobalTensor<float> leftSingularMatrix = qtGm[leftSubMatrix.start_col * LDN + leftSubMatrix.start_col];
        GlobalTensor<float> rightSingularMatrix = wtGm[leftSubMatrix.start_col * LDN + leftSubMatrix.start_col];
        GlobalTensor<float> st = stGm[leftSubMatrix.start_col * LDN + leftSubMatrix.start_col];
        GlobalTensor<float> gt = gtGm[leftSubMatrix.start_col * LDN + leftSubMatrix.start_col];
        auto tmpSpace = tmpGm[leftSubMatrix.start_col * LDN + leftSubMatrix.start_col];
        GlobalTensor<float> f = fGm[leftSubMatrix.start_col];
        GlobalTensor<float> l = lGm[leftSubMatrix.start_col];
        GlobalTensor<uint32_t> idxq = idxqGm[leftSubMatrix.start_col];
        auto idxc = idxcGm[leftSubMatrix.start_col];
        auto idxp = idxpGm[leftSubMatrix.start_col];
        auto idx = idxGm[leftSubMatrix.start_col];
        auto coltyp = coltypGm[leftSubMatrix.start_col];
        auto dsigma = dsigmaGm[leftSubMatrix.start_col];
        // scale for stability,make later deflation tol calculation feasible
        float orgnrm = max(fabs(alpha), fabs(beta));
        uint16_t totalRowNum = leftColNum + rightColNum - 1 + isSquare;
        d(leftColNum - 1) = 0.0f;
        for (auto i = 0; i < totalRowNum; i++)
        {
            if (fabs(d(i)) > orgnrm)
            {
                orgnrm = fabs(d(i));
            }
        }
        // TODO orgnrm 为0时，特殊处理
        alpha /= orgnrm;
        beta /= orgnrm;
        { // scale d by orgnrm
            RefreshAllCache();
            const DataCopyExtParams copyInParams = {1, totalRowNum * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParams = {true, 0, 0, 0.0f};
            const DataCopyExtParams copyOutParams = {1, totalRowNum * sizeOfFloat, 0, 0, 0};
            LocalTensor<float> inputTensor = inQueue.AllocTensor<float>();
            DataCopyPad(inputTensor, d, copyInParams, copyInPadParams);
            inQueue.EnQue(inputTensor);

            inputTensor = inQueue.DeQue<float>();
            auto outputTensor = outQueue.AllocTensor<float>();
            Muls<float>(outputTensor, inputTensor, 1.0f / orgnrm, totalRowNum);
            inQueue.FreeTensor(inputTensor);
            outQueue.EnQue(outputTensor);

            outputTensor = outQueue.DeQue<float>();
            DataCopyPad(d, outputTensor, copyOutParams);
            outQueue.FreeTensor(outputTensor);
        }

        // get idxq
        // sort d and get another permutation
        // deflate z1
        // deflate z and d
        // rotate to remove the N+1th column if necessary
        // permute qt and wt
        printf("[MergeSubMatrix] 调用Deflation前\n");
        Deflation(leftColNum - 1, rightColNum - 1 + isSquare, isSquare, beta, alpha, k, d, z, leftSingularMatrix, rightSingularMatrix, st, gt, f, l, idxq, idxc, idxp, idx, coltyp, dsigma);
        printf("[MergeSubMatrix] Deflation完成, k=%d\n", k);
        // call secular quation solver to get sigma and singular vectors
        // update singular vectors with matmul
        // sort sigma and form idxq
        printf("[MergeSubMatrix] MergeSubMatrix_step2前\n");
        MergeSubMatrix_step2(k, leftColNum, rightColNum, isSquare, leftSingularMatrix, rightSingularMatrix, d, st, gt, f, l, idxq, idxc, coltyp, dsigma, z, tmpSpace);
        printf("[MergeSubMatrix] MergeSubMatrix_step2完成\n");
        { // unscale d
            RefreshAllCache();
            const DataCopyExtParams copyInParams = {1, totalRowNum * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParams = {true, 0, 0, 0.0f};
            const DataCopyExtParams copyOutParams = {1, totalRowNum * sizeOfFloat, 0, 0, 0};
            LocalTensor<float> inputTensor = inQueue.AllocTensor<float>();
            DataCopyPad(inputTensor, d, copyInParams, copyInPadParams);
            inQueue.EnQue(inputTensor);

            inputTensor = inQueue.DeQue<float>();
            auto outputTensor = outQueue.AllocTensor<float>();
            Muls<float>(outputTensor, inputTensor, orgnrm, totalRowNum);
            inQueue.FreeTensor(inputTensor);
            outQueue.EnQue(outputTensor);

            outputTensor = outQueue.DeQue<float>();
            DataCopyPad(d, outputTensor, copyOutParams);
            outQueue.FreeTensor(outputTensor);
        }
        printf("[MergeSubMatrix] 完成\n");

        return;
    }

    __aicore__ inline void Deflation(uint16_t leftRowNum, uint16_t rightRowNum, bool isSquare, float beta, float alpha, uint16_t &k, GlobalTensor<float> &d, GlobalTensor<float> &z, GlobalTensor<float> &leftSingularMatrix, GlobalTensor<float> &rightSingularMatrix, GlobalTensor<float> &st, GlobalTensor<float> &gt, GlobalTensor<float> &f, GlobalTensor<float> &l, GlobalTensor<uint32_t> &idxq, GlobalTensor<uint32_t> &idxc, GlobalTensor<uint32_t> &idxp, GlobalTensor<uint32_t> &idx, GlobalTensor<uint32_t> &coltyp, GlobalTensor<float> &dsigma)
    {
        printf("[Deflation] leftRowNum=%d, rightRowNum=%d, isSquare=%d, beta=%f, alpha=%f\n", leftRowNum, rightRowNum, isSquare, beta, alpha);
        const uint16_t totalRowNum = leftRowNum + rightRowNum + 1, totalColNum = totalRowNum + 1 - isSquare, dNum = totalRowNum - 1;
        const uint16_t leftRowNump1 = leftRowNum + 1;
        const uint16_t leftRowNump2 = leftRowNum + 2;
        const uint16_t leftColNum = leftRowNum + 1;
        const uint16_t rightColNum = rightRowNum + 1 - isSquare;
        printf("[Deflation] dNum=%d, totalRowNum=%d, totalColNum=%d\n", dNum, totalRowNum, totalColNum);
        // form d and z
        RefreshAllCache();
        LocalTensor<float> inputTensor, outputTensor, bindLocalf;
        LocalTensor<uint32_t> indexTensor, outIndexTensor;

        // the first part
        // the first part of z is (lamba,l[0:leftRowNum])*alpha
        {
            const float lambda1 = l(leftRowNum);
            const DataCopyExtParams copyInParams = {1, leftRowNum * sizeOfFloat, 0, 0, 0};
            const DataCopyExtParams copyOutParams = {1, leftRowNump1 * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParams = {true, 1, 0, lambda1};
            inputTensor = inQueue.AllocTensor<float>();
            DataCopyPad(inputTensor, l, copyInParams, copyInPadParams);
            inQueue.EnQue(inputTensor);

            inputTensor = inQueue.DeQue<float>();
            outputTensor = outQueue.AllocTensor<float>();
            Muls(outputTensor, inputTensor, alpha, leftRowNump1);
            inQueue.FreeTensor(inputTensor);
            outQueue.EnQue(outputTensor);

            outputTensor = outQueue.DeQue<float>();
            DataCopyPad(z, outputTensor, copyOutParams);
            outQueue.FreeTensor(outputTensor);
        }
        // shift the first part of d right by 1,d[i+1] = d[i]
        {
            const DataCopyExtParams copyInParams = {1, leftRowNum * sizeOfFloat, 0, 0, 0};
            const DataCopyExtParams copyOutParams = {1, leftRowNump1 * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParams = {true, 1, 0, 0.0f};
            bindLocalf = copyBind.AllocTensor<float>();
            DataCopyPad(bindLocalf, d, copyInParams, copyInPadParams);
            copyBind.EnQue(bindLocalf);
            bindLocalf = copyBind.DeQue<float>();
            DataCopyPad(d, bindLocalf, copyOutParams);
            copyBind.FreeTensor(bindLocalf);
        }

        // construct the first part of idxq ,idxq[i+1] = idxq[i]+1
        {
            // 因为矢量计算不支持uint32_t，改成int32_t又类型不兼容.reinterpretCast
            const DataCopyExtParams copyInParams = {1, leftRowNum * sizeOfUint32_t, 0, 0, 0};
            const DataCopyExtParams copyOutParams = {1, leftRowNum * sizeOfUint32_t, 0, 0, 0};
            const DataCopyPadExtParams<uint32_t> copyInPadParams = {true, 0, 0, 0};
            indexTensor = inQueue.AllocTensor<uint32_t>();
            DataCopyPad(indexTensor, idxq, copyInParams, copyInPadParams);
            inQueue.EnQue(indexTensor);

            indexTensor = inQueue.DeQue<uint32_t>();
            outIndexTensor = outQueue.AllocTensor<uint32_t>();
            Adds<int32_t>(outIndexTensor.ReinterpretCast<int32_t>(), indexTensor.ReinterpretCast<int32_t>(), 1, leftRowNump1);
            inQueue.FreeTensor(indexTensor);
            outQueue.EnQue(outIndexTensor);

            outIndexTensor = outQueue.DeQue<uint32_t>();
            DataCopyPad(idxq[1], outIndexTensor, copyOutParams);
            outQueue.FreeTensor(outIndexTensor);
        }
        // the second part of d z and idxq
        // z[i]=f[i]*beta
        // leftRowNump1 is also the shift of the second part
        {
            const DataCopyExtParams copyInParams = {1, rightColNum * sizeOfFloat, 0, 0, 0};
            const DataCopyExtParams copyOutParams = {1, rightColNum * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParams = {true, 0, 0, 0};
            inputTensor = inQueue.AllocTensor<float>();
            DataCopyPad(inputTensor, f[leftRowNump1], copyInParams, copyInPadParams);
            inQueue.EnQue(inputTensor);

            inputTensor = inQueue.DeQue<float>();
            outputTensor = outQueue.AllocTensor<float>();
            Muls(outputTensor, inputTensor, beta, rightColNum);
            inQueue.FreeTensor(inputTensor);
            outQueue.EnQue(outputTensor);

            outputTensor = outQueue.DeQue<float>();
            DataCopyPad(z[leftRowNump1], outputTensor, copyOutParams);
            outQueue.FreeTensor(outputTensor);
        }

        // no need to shift d.construct the second part of idxq
        {
            const DataCopyExtParams copyInParams = {1, rightRowNum * sizeOfUint32_t, 0, 0, 0};
            const DataCopyExtParams copyOutParams = {1, rightRowNum * sizeOfUint32_t, 0, 0, 0};
            const DataCopyPadExtParams<uint32_t> copyInPadParams = {true, 0, 0, 0};
            indexTensor = inQueue.AllocTensor<uint32_t>();
            DataCopyPad(indexTensor, idxq[leftRowNump1], copyInParams, copyInPadParams);
            inQueue.EnQue(indexTensor);

            indexTensor = inQueue.DeQue<uint32_t>();
            outIndexTensor = outQueue.AllocTensor<uint32_t>();
            Adds<int32_t>(outIndexTensor.ReinterpretCast<int32_t>(), indexTensor.ReinterpretCast<int32_t>(), leftRowNump1, rightRowNum);
            inQueue.FreeTensor(indexTensor);
            outQueue.EnQue(outIndexTensor);

            outIndexTensor = outQueue.DeQue<uint32_t>();
            DataCopyPad(idxq[leftRowNump1], outIndexTensor, copyOutParams);
            outQueue.FreeTensor(outIndexTensor);
        }

        // init coltype,0 and 1
        // {
        //     const DataCopyExtParams copyOutParams = {1, leftRowNum * sizeOfUint32_t, 0, 0, 0};
        //     outIndexTensor = outQueue.AllocTensor<uint32_t>();
        //     Duplicate<uint32_t>(outIndexTensor, 0, leftRowNum);
        //     outQueue.EnQue(outIndexTensor);

        //     outIndexTensor = outQueue.DeQue<uint32_t>();
        //     DataCopyPad(coltyp[1], outIndexTensor, copyOutParams);
        //     outQueue.FreeTensor(outIndexTensor);
        // }
        // {
        //     const DataCopyExtParams copyOutParams = {1, rightRowNum * sizeOfUint32_t, 0, 0, 0};
        //     outIndexTensor = outQueue.AllocTensor<uint32_t>();
        //     Duplicate<uint32_t>(outIndexTensor, 1, rightRowNum);
        //     outQueue.EnQue(outIndexTensor);

        //     outIndexTensor = outQueue.DeQue<uint32_t>();
        //     DataCopyPad(coltyp[leftRowNump1], outIndexTensor, copyOutParams);
        //     outQueue.FreeTensor(outIndexTensor);
        // }
        // init tmp
        {
            const DataCopyExtParams copyInParamsi = {1, dNum * sizeOfUint32_t, 0, 0, 0};
            const DataCopyPadExtParams<uint32_t> copyInPadParamsi = {true, 0, 0, 0};
            indexTensor = inQueue.AllocTensor<uint32_t>();
            DataCopyPad(indexTensor, idxq[1], copyInParamsi, copyInPadParamsi);
            inQueue.EnQue(indexTensor);

            indexTensor = inQueue.DeQue<uint32_t>();
            Muls<int32_t>(indexTensor.ReinterpretCast<int32_t>(), indexTensor.ReinterpretCast<int32_t>(), sizeOfFloat, dNum);

            // get dsigma
            const DataCopyExtParams copyInParamsf = {1, totalRowNum * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParamsf = {true, 0, 0, 0.0f};
            const DataCopyExtParams copyOutParams = {1, dNum * sizeOfFloat, 0, 0, 0};
            {
                inputTensor = inQueue.AllocTensor<float>();
                DataCopyPad(inputTensor, d, copyInParamsf, copyInPadParamsf);
                inQueue.EnQue(inputTensor);

                inputTensor = inQueue.DeQue<float>();
                outputTensor = outQueue.AllocTensor<float>();
                // get bytes offset
                Gather(outputTensor, inputTensor, indexTensor, 0, dNum);
                // singleDumpTensor(inputTensor, 64);
                // singleDumpTensor(indexTensor, 64);
                // singleDumpTensor(outputTensor, 64);
                inQueue.FreeTensor(inputTensor);
                outQueue.EnQue(outputTensor);

                outputTensor = outQueue.DeQue<float>();
                DataCopyPad(dsigma, outputTensor, copyOutParams);
                outQueue.FreeTensor(outputTensor);
            }
            // get z sorted by idxq
            {
                inputTensor = inQueue.AllocTensor<float>();
                DataCopyPad(inputTensor, z, copyInParamsf, copyInPadParamsf);
                inQueue.EnQue(inputTensor);

                inputTensor = inQueue.DeQue<float>();
                outputTensor = outQueue.AllocTensor<float>();
                Gather(outputTensor, inputTensor, indexTensor, 0, dNum);
                inQueue.FreeTensor(inputTensor);
                outQueue.EnQue(outputTensor);

                outputTensor = outQueue.DeQue<float>();
                DataCopyPad(st, outputTensor, copyOutParams);
                outQueue.FreeTensor(outputTensor);
            }

            // get coltyp sorted by idxq
            // {
            //     auto inputTensor = inQueue.AllocTensor<uint32_t>();
            //     DataCopyPad(inputTensor, coltyp, copyInParamsi, copyInPadParamsi);
            //     inQueue.EnQue(inputTensor);

            //     inputTensor = inQueue.DeQue<uint32_t>();
            //     auto outputTensor = outQueue.AllocTensor<uint32_t>();
            //     Gather(outputTensor, inputTensor, indexTensor, 0, dNum);
            //     inQueue.FreeTensor(inputTensor);
            //     outQueue.EnQue(outputTensor);

            //     outputTensor = outQueue.DeQue<uint32_t>();
            //     DataCopyPad(idxc, outputTensor, copyOutParams);
            //     outQueue.FreeTensor(outputTensor);
            // }

            inQueue.FreeTensor(indexTensor);
        }

        // sort dsigma ,get idx
        {
            // firstly construct MrgSort list.4B score 4B index.8n,8n+4
            //because Scatter is not supported in AICore,so we need to construct the mrgSort list by scalar operation
            //the dsigma is multiple by -1 to make it descending order
            {
                constructMrgSortList(tmpBuf1.Get<float>(), dsigma, dNum);
            }
            // MrgSort the mrglist
            printf("[Deflation] MrgSort the mrglist,leftRowNum*2=%d\n", leftRowNum * 2);
            {
                LocalTensor<float> workTensor = tmpBuf1.Get<float>();
                LocalTensor<float> dstTensor = tmpBuf2.Get<float>();
                MrgSortSrcList<float> mrgSortSrcList;
                mrgSortSrcList.src1 = workTensor;
                mrgSortSrcList.src2 = workTensor[leftRowNum * 2];
                printf("[Deflation] mrgSortSrcList.src1=%p,mrgSortSrcList.src2=%p\n", mrgSortSrcList.src1.GetPhyAddr(), mrgSortSrcList.src2.GetPhyAddr());
                MrgSort4Info params;
                params.elementLengths[0] = leftRowNum;
                params.elementLengths[1] = rightRowNum;
                params.ifExhaustedSuspension = false;
                params.validBit = 3;
                params.repeatTimes = 1;
                singleDumpTensor(workTensor, 64);
                singleDumpTensor(workTensor.ReinterpretCast<uint32_t>(), 64);
                MrgSort(dstTensor, mrgSortSrcList, params);
                singleDumpTensor(dstTensor, 64);
                singleDumpTensor(dstTensor.ReinterpretCast<uint32_t>(), 64);
            }
            // get the sorted d and idx
            {
                const DataCopyExtParams copyOutParamsf = {1, dNum * sizeOfFloat, 0, 0, 0};
                const DataCopyExtParams copyOutParamsi = {1, dNum * sizeOfUint32_t, 0, 0, 0};

                LocalTensor<float> mrglistf = tmpBuf2.Get<float>();
                LocalTensor<uint32_t> mrglisti = tmpBuf2.Get<uint32_t>();

                indexTensor = inQueue.AllocTensor<uint32_t>();
                outputTensor = outQueue.AllocTensor<float>();
                outIndexTensor = outQueue.AllocTensor<uint32_t>();
                CreateVecIndex<int32_t>(indexTensor.ReinterpretCast<int32_t>(), 0, dNum);
                Muls<int32_t>(indexTensor.ReinterpretCast<int32_t>(), indexTensor.ReinterpretCast<int32_t>(), 8, dNum);
                Gather(outputTensor, mrglistf, indexTensor, 0, dNum);
                // 因为之前乘了-1来归并排序，所以这里需要乘以-1变成升序
                Muls<float>(outputTensor, outputTensor, -1, dNum);
                Gather(outIndexTensor, mrglisti, indexTensor, 4, dNum);
                outQueue.EnQue(outputTensor);
                outQueue.EnQue(outIndexTensor);
                inQueue.FreeTensor(indexTensor);

                outputTensor = outQueue.DeQue<float>();
                DataCopyPad(d[1], outputTensor, copyOutParamsf);
                outQueue.FreeTensor(outputTensor);

                outIndexTensor = outQueue.DeQue<uint32_t>();
                DataCopyPad(idx[1], outIndexTensor, copyOutParamsi);
                outQueue.FreeTensor(outIndexTensor);
            }
        }
        printf("[Deflation] after sort dsigma,d,idx\n");
        // use idx to permute d z and coltype.d has been permuted
        {
            const DataCopyExtParams copyInParamsi = {1, dNum * sizeOfUint32_t, 0, 0, 0};
            const DataCopyPadExtParams<uint32_t> copyInPadParamsi = {true, 0, 0, 0};
            indexTensor = inQueue.AllocTensor<uint32_t>();
            DataCopyPad(indexTensor, idx[1], copyInParamsi, copyInPadParamsi);
            inQueue.EnQue(indexTensor);

            indexTensor = inQueue.DeQue<uint32_t>();
            Muls<int32_t>(indexTensor.ReinterpretCast<int32_t>(), indexTensor.ReinterpretCast<int32_t>(), sizeOfFloat, dNum);

            // permute z
            const DataCopyExtParams copyInParamsf = {1, dNum * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParamsf = {true, 0, 0, 0.0f};
            const DataCopyExtParams copyOutParamsf = {1, dNum * sizeOfFloat, 0, 0, 0};
            inputTensor = inQueue.AllocTensor<float>();
            DataCopyPad(inputTensor, st, copyInParamsf, copyInPadParamsf);
            inQueue.EnQue(inputTensor);

            inputTensor = inQueue.DeQue<float>();
            outputTensor = outQueue.AllocTensor<float>();
            Gather(outputTensor, inputTensor, indexTensor, 0, dNum);
            outQueue.EnQue(outputTensor);
            inQueue.FreeTensor(inputTensor);

            outputTensor = outQueue.DeQue<float>();
            DataCopyPad(z[1], outputTensor, copyOutParamsf);
            outQueue.FreeTensor(outputTensor);

            // permute coltyp
            // const DataCopyExtParams copyOutParamsi = {1, dNum * sizeOfUint32_t, 0, 0, 0};
            // auto coltypTensor = inQueue.AllocTensor<uint32_t>();
            // DataCopyPad(coltypTensor, idxc, copyInParamsi, copyInPadParamsi);
            // inQueue.EnQue(coltypTensor);

            // coltypTensor = inQueue.DeQue<uint32_t>();
            // auto outColtypTensor = outQueue.AllocTensor<uint32_t>();
            // Gather(outColtypTensor, coltypTensor, indexTensor, 0, dNum);
            // outQueue.EnQue(outColtypTensor);
            // inQueue.FreeTensor(coltypTensor);

            // outColtypTensor = outQueue.DeQue<uint32_t>();
            // DataCopyPad(coltyp[1], outColtypTensor, copyOutParamsi);
            // outQueue.FreeTensor(outColtypTensor);

            // 释放 indexTensor
            inQueue.FreeTensor(indexTensor);
        }

        // calc tol
        float tol = max(fabs(alpha), fabs(beta));
        tol = 8 * EPS * max(fabs(d(dNum)), tol); // the totalRowNum th diagonal
        k = 0;
        uint16_t k2 = totalRowNum, jprev, j, idxjp, idxj;
        float c, s, tau;
        printf("[Deflation] before find the first undeflated z\n");
        // find the first undeflated z
        for (j = 1; j < totalRowNum; ++j)
        {
            if (fabs(z(j)) <= tol)
            {
                --k2;
                idxp(k2) = j;
                // coltyp(j) = 3;
                if (j == totalRowNum - 1)
                {
                    goto label120;
                }
            }
            else
            {
                jprev = j;
                break;
            }
        }
        j = jprev;
        while (1)
        {
            j = j + 1;
            if (j >= totalRowNum)
            {
                break;
            }
            if (fabs(z(j)) <= tol)
            {
                // deflate the z
                --k2;
                idxp(k2) = j;
                // coltyp(j) = 3;
            }
            else
            {
                // check whether can be deflated due to d
                if (fabs(d(j) - d(jprev)) <= tol)
                {
                    // deflate due to close d
                    s = z(jprev);
                    c = z(j);
                    tau = sqrt(c * c + s * s);
                    c = c / tau;
                    s = -s / tau;
                    // deflate the jprev,use j as new jprev
                    z(j) = tau;
                    z(jprev) = 0;
                    // update the left singular matrix
                    idxjp = idxq(idx(jprev) + 1);
                    idxj = idxq(idx(j) + 1);
                    if (idxjp < leftRowNump1)
                    {
                        --idxjp;
                    }
                    if (idxj < leftRowNump1)
                    {
                        --idxj;
                    }
                    // rotate leftSingularMatrix and rightSingularMatrix
                    // row jp=c rowjp+s rowj
                    // row j= -s rowjp+c rowj
                    rotateRow(totalRowNum, leftSingularMatrix[idxjp * LDN], leftSingularMatrix[idxj * LDN], c, s);
                    rotateRow(totalColNum, rightSingularMatrix[idxjp * LDN], rightSingularMatrix[idxj * LDN], c, s);

                    // if (coltyp(j) != coltyp(jprev))
                    // {
                    //     coltyp(j) = 2;
                    // }
                    // coltyp(jprev) = 3;
                    --k2;
                    idxp(k2) = jprev;
                    jprev = j;
                }
                else
                {
                    // record the iprev as undeflated,and update jprev
                    ++k;
                    // begin from 1.
                    // st(k) = z(jprev);
                    // dsigma(k) = d(jprev);
                    idxp(k) = jprev;
                    jprev = j;
                }
            }
        }
        // record the last singular value
        ++k;
        // st(k) = z(jprev);
        // dsigma(k) = d(jprev);
        idxp(k) = jprev;
    label120:
        // not using ctot and psm,to simplify the code and take advantage of AIC
        RefreshAllCache();
        // sort the singular values into dsigma
        printf("[Deflation] before sort singular values into dsigma\n");
        {
            const DataCopyExtParams copyInParamsf = {1, dNum * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParamsf = {true, 0, 0, 0.0f};
            const DataCopyExtParams copyOutParamsf = {1, dNum * sizeOfFloat, 0, 0, 0};
            const DataCopyExtParams copyInParamsi = {1, dNum * sizeOfUint32_t, 0, 0, 0};
            const DataCopyPadExtParams<uint32_t> copyInPadParamsi = {true, 0, 0, 0};
            auto dTensor = inQueue.AllocTensor<float>();
            DataCopyPad(dTensor, d[1], copyInParamsf, copyInPadParamsf);
            inQueue.EnQue(dTensor);

            auto idxpTensor = inQueue.AllocTensor<uint32_t>();
            DataCopyPad(idxpTensor, idxp[1], copyInParamsi, copyInPadParamsi); // idxp start from 1,so it needs to minus 1
            inQueue.EnQue(idxpTensor);

            dTensor = inQueue.DeQue<float>();
            idxpTensor = inQueue.DeQue<uint32_t>();
            outputTensor = outQueue.AllocTensor<float>();
            Adds<int32_t>(idxpTensor.ReinterpretCast<int32_t>(), idxpTensor.ReinterpretCast<int32_t>(), -1, dNum);
            Muls<int32_t>(idxpTensor.ReinterpretCast<int32_t>(), idxpTensor.ReinterpretCast<int32_t>(), sizeOfFloat, dNum);
            Gather(outputTensor, dTensor, idxpTensor, 0, dNum);
            outQueue.EnQue(outputTensor);
            inQueue.FreeTensor(dTensor);
            inQueue.FreeTensor(idxpTensor);

            outputTensor = outQueue.DeQue<float>();
            DataCopyPad(dsigma[1], outputTensor, copyOutParamsf);
            outQueue.FreeTensor(outputTensor);
        }

        printf("[Deflation] before sort corresponding singular vectors into st,gt\n");
        // sort  corresponding singular vectors into  st, and gt respectively.
        for (j = 1; j < totalRowNum; ++j)
        {
            uint32_t jp = idxp(j);
            dsigma(j) = d(jp);
            uint32_t idxj = idxq(idx(idxp(j)) + 1);
            if (idxj < leftRowNump1)
            {
                --idxj;
            }
            // copy the idxj-th row to jth row of st and gt
            copyRow(totalRowNum, leftSingularMatrix[idxj * LDN], st[j * LDN]);
            copyRow(totalColNum, rightSingularMatrix[idxj * LDN], gt[j * LDN]);
        }
        printf("[Deflation] after sort corresponding singular vectors into st,gt\n");
        // Determine dsigma(0),dsigma(1),z(0)
        {
            dsigma(0) = 0.0f;
            float hlftol = tol / 2.0f;
            if (fabs(dsigma(1)) <= hlftol)
            {
                dsigma(1) = hlftol;
            }
            // notSquare,rotate another time
            if (totalColNum > totalRowNum)
            {
                // rotate another time
                float z0 = z(0);
                z(0) = sqrt(z0 * z0 + z(totalRowNum) * z(totalRowNum));
                if (z(0) <= tol)
                {
                    // z0 and z(totalRowNum) are both small,so we deflate both without rotation
                    c = 1;
                    s = 0;
                    z(0) = tol;
                }
                else
                {
                    c = z0 / z(0);
                    s = z(totalRowNum) / z(0);
                }
            }
            else
            {
                if (fabs(z(0)) <= tol)
                {
                    z(0) = tol;
                }
                else
                {
                    // z(0)=z(0)
                }
            }
        }

        ++k; // k functions as index before,now it functions as the number of nondeflated cols

        printf("[Deflation] before sort z by idxp\n");
        // sort z by idxp
        {
            const DataCopyExtParams copyInParamsf = {1,  (k - 1) * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParamsf = {true, 0, 0, 0.0f};
            const DataCopyExtParams copyOutParamsf = {1, (k - 1) * sizeOfFloat, 0, 0, 0};
            const DataCopyExtParams copyInParamsi = {1, (k - 1) * sizeOfUint32_t, 0, 0, 0};
            const DataCopyPadExtParams<uint32_t> copyInPadParamsi = {true, 0, 0, 0};
            // 用idxp来gather z[1:]到z[1:]
            // 首先获取idxp
            indexTensor = inQueue.AllocTensor<uint32_t>();
            DataCopyPad(indexTensor, idxp[1], copyInParamsi, copyInPadParamsi);
            inQueue.EnQue(indexTensor);

            // 获取z
            inputTensor = inQueue.AllocTensor<float>();
            DataCopyPad(inputTensor, z[1], copyInParamsf, copyInPadParamsf);
            inQueue.EnQue(inputTensor);

            indexTensor = inQueue.DeQue<uint32_t>();
            inputTensor = inQueue.DeQue<float>();
            outputTensor = outQueue.AllocTensor<float>();
            Adds<int32_t>(indexTensor.ReinterpretCast<int32_t>(), indexTensor.ReinterpretCast<int32_t>(), -1, k - 1);
            Muls<int32_t>(indexTensor.ReinterpretCast<int32_t>(), indexTensor.ReinterpretCast<int32_t>(), sizeOfFloat, k - 1);
            Gather(outputTensor, inputTensor, indexTensor, 0, k - 1);
            outQueue.EnQue(outputTensor);

            // 释放输入张量
            inQueue.FreeTensor(inputTensor);
            inQueue.FreeTensor(indexTensor);

            // 获取结果并写回z
            outputTensor = outQueue.DeQue<float>();
            DataCopyPad(z[1], outputTensor, copyOutParamsf);
            outQueue.FreeTensor(outputTensor);
        }

        // set the first rows of st and gt,and last row of gt
        printf("[Deflation] before set the first rows of st\n");
        // first row of st
        {
            const DataCopyExtParams copyOutParamsf = {1, totalRowNum * sizeOfFloat, 0, 0, 0};
            outputTensor = outQueue.AllocTensor<float>();
            Duplicate(outputTensor, 0.0f, totalRowNum);
            outQueue.EnQue(outputTensor);
            outputTensor = outQueue.DeQue<float>();
            DataCopyPad(st, outputTensor, copyOutParamsf);
            outQueue.FreeTensor(outputTensor);
            st(leftRowNum) = 1.0f;
        }
        printf("[Deflation] before set the first rows of gt\n");
        // first row and last row of gt
        {
            if (totalColNum > totalRowNum)
            {
                // first row of gt is c rightSingularMatrixRow[LeftRowNum]+s rightSingularMatrixRow[totalColNum-1]
                // last row of gt is -s rightSingularMatrixRow[LeftRowNum]+c rightSingularMatrixRow[totalColNum-1]
                {
                    const DataCopyExtParams copyInParamsf = {1, leftColNum * sizeOfFloat, 0, 0, 0};
                    const DataCopyExtParams copyOutParamsf = {1, leftColNum * sizeOfFloat, 0, 0, 0};
                    const DataCopyPadExtParams<float> copyInPadParamsf = {true, 0, 0, 0.0f};
                    inputTensor = inQueue.AllocTensor<float>();
                    DataCopyPad(inputTensor, rightSingularMatrix[leftRowNum * LDN], copyInParamsf, copyInPadParamsf);
                    inQueue.EnQue(inputTensor);

                    inputTensor = inQueue.DeQue<float>();
                    outputTensor = outQueue.AllocTensor<float>();
                    auto outputTensor2 = outQueue.AllocTensor<float>();
                    Muls(outputTensor, inputTensor, c, leftColNum);
                    Muls(outputTensor2, inputTensor, -s, leftColNum);
                    inQueue.FreeTensor(inputTensor);
                    outQueue.EnQue(outputTensor);
                    outQueue.EnQue(outputTensor2);

                    outputTensor = outQueue.DeQue<float>();
                    outputTensor2 = outQueue.DeQue<float>();
                    DataCopyPad(gt, outputTensor, copyOutParamsf);
                    DataCopyPad(gt[(totalColNum - 1) * LDN], outputTensor2, copyOutParamsf);
                    outQueue.FreeTensor(outputTensor);
                    outQueue.FreeTensor(outputTensor2);
                }
                {
                    const DataCopyExtParams copyInParamsf = {1, rightColNum * sizeOfFloat, 0, 0, 0};
                    const DataCopyExtParams copyOutParamsf = {1, rightColNum * sizeOfFloat, 0, 0, 0};
                    const DataCopyPadExtParams<float> copyInPadParamsf = {true, 0, 0, 0.0f};
                    inputTensor = inQueue.AllocTensor<float>();
                    DataCopyPad(inputTensor, rightSingularMatrix[(totalColNum - 1) * LDN + leftColNum], copyInParamsf, copyInPadParamsf);
                    inQueue.EnQue(inputTensor);

                    inputTensor = inQueue.DeQue<float>();
                    outputTensor = outQueue.AllocTensor<float>();
                    auto outputTensor2 = outQueue.AllocTensor<float>();
                    Muls(outputTensor, inputTensor, s, rightColNum);
                    Muls(outputTensor2, inputTensor, c, rightColNum);
                    inQueue.FreeTensor(inputTensor);
                    outQueue.EnQue(outputTensor);
                    outQueue.EnQue(outputTensor2);

                    outputTensor = outQueue.DeQue<float>();
                    outputTensor2 = outQueue.DeQue<float>();
                    DataCopyPad(gt[leftColNum], outputTensor, copyOutParamsf);
                    DataCopyPad(gt[(totalColNum - 1) * LDN + leftColNum], outputTensor2, copyOutParamsf);
                    outQueue.FreeTensor(outputTensor);
                    outQueue.FreeTensor(outputTensor2);
                }
                // copy the last row of gt to the last row of rightSingularMatrix and form l and f
                copyRow(totalColNum, gt[(totalColNum - 1) * LDN], rightSingularMatrix[(totalColNum - 1) * LDN]);
                f(totalColNum - 1) = gt((totalColNum - 1) * LDN);
                l(totalColNum - 1) = gt((totalColNum - 1) * LDN + totalColNum - 1);
            }
            else
            {
                // is square
                //  copy rightSingularMatrixRow[leftRowNum] to gt[0]
                //  no need to compute the last row of gt
                copyRow(totalColNum, rightSingularMatrix[leftRowNum * LDN], gt);
            }
        }

        printf("[Deflation] before there's deflation,copy the decided singular values and vectors back,totalRowNum=%d,k=%d\n", totalRowNum, k);
        if (totalRowNum > k)
        {
            // there's deflation,copy the decided singular values and vectors back
            // dsigma is computed by scalar,refreshAllCashe first
            RefreshAllCache();
            copyRow(totalRowNum - k, dsigma[k], d[k]);
            // copy totalRowNum-k rows of left singular vects and right singular vects
            // copy st/gt to leftSingularMatrix/rightSingularMatrix
            // update the corresponding f and l
            for (int j = k; j < totalRowNum; ++j)
            {
                copyRow(totalRowNum, st[j * LDN], leftSingularMatrix[j * LDN]);
                copyRow(totalColNum, gt[j * LDN], rightSingularMatrix[j * LDN]);
                f(j) = gt(j * LDN);
                l(j) = gt(j * LDN + totalColNum - 1);
            }
        }
        
        return;
    }

    __aicore__ inline void SecularEquationSolver(const uint16_t n, const uint16_t i, const GlobalTensor<float> &dsigma, const GlobalTensor<float> &z, const GlobalTensor<float> &tmpSpace, const GlobalTensor<float> &d)
    {
        printf("[SecularEquationSolver] n=%d, i=%d\n", n, i);
        LocalTensor<float> inputTensor, outputTensor;
        float miu, omega, di, dip1, dChosen, di2, dip12;
        // actual d stored in dsigma,tmp1 stores delta dj-di or dj-dip1,tmp2 stores dj+di or dj+dip1
        if (i != n - 1)
        {
            float psi1, psi2, c1, c2, c3, psi1derivative, psi2derivative, c1hat, c2hat, result, tol;
            const uint16_t leftNums = i + 1, rightNums = n - i - 1;
            const uint16_t leftUpperLength = (leftNums + BlockFloatCnt - 1) / BlockFloatCnt * BlockFloatCnt;
            const uint16_t rightUpperLength = (rightNums + BlockFloatCnt - 1) / BlockFloatCnt * BlockFloatCnt;
            // tmpBuf1 is for delta and work
            LocalTensor<float> leftDelta = tmpBuf1.Get<float>();
            LocalTensor<float> rightDelta = leftDelta[leftUpperLength];
            LocalTensor<float> leftWork = rightDelta[rightUpperLength];
            LocalTensor<float> rightWork = leftWork[leftUpperLength];
            // tmpBuf2 is for z and otherTmp
            LocalTensor<float> leftZ2 = tmpBuf2.Get<float>();
            LocalTensor<float> rightZ2 = leftZ2[leftUpperLength];
            LocalTensor<float> leftOtherTmp = rightZ2[rightUpperLength];
            LocalTensor<float> rightOtherTmp = leftOtherTmp[leftUpperLength];

            const DataCopyExtParams copyInParams1 = {1, leftNums * sizeOfFloat, 0, 0, 0};
            const DataCopyExtParams copyOutParams1 = {1, leftNums * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParams1 = {true, 0, 0, 0.0f};
            const DataCopyExtParams copyInParams2 = {1, rightNums * sizeOfFloat, 0, 0, 0};
            const DataCopyExtParams copyOutParams2 = {1, rightNums * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParams2 = {true, 0, 0, 0.0f};
            di = dsigma(i);
            dip1 = dsigma(i + 1);
            di2 = di * di;
            dip12 = dip1 * dip1;
            omega = (di + dip1) / 2;
            // decide whether use dj+di or dj+dip1 and the definition of delta
            result = 0.0f;
            // init  delta,work to d,z2 to z^2
            {
                inputTensor = inQueue.AllocTensor<float>();
                auto inputTensor2 = inQueue.AllocTensor<float>();
                DataCopyPad(inputTensor, dsigma, copyInParams1, copyInPadParams1);
                DataCopyPad(inputTensor2, dsigma[leftNums], copyInParams2, copyInPadParams2);
                inQueue.EnQue(inputTensor);
                inQueue.EnQue(inputTensor2);

                inputTensor = inQueue.DeQue<float>();
                inputTensor2 = inQueue.DeQue<float>();
                // DataCopy requires 32B align,use Adds instead
                Adds<float>(leftDelta, inputTensor, 0.0f, leftNums);
                Adds<float>(leftWork, inputTensor, 0.0f, leftNums);
                Adds<float>(rightDelta, inputTensor2, 0.0f, rightNums);
                Adds<float>(rightWork, inputTensor2, 0.0f, rightNums);
                inQueue.FreeTensor(inputTensor);
                inQueue.FreeTensor(inputTensor2);
                // init z2 to z^2
                inputTensor = inQueue.AllocTensor<float>();
                inputTensor2 = inQueue.AllocTensor<float>();
                DataCopyPad(inputTensor, z, copyInParams1, copyInPadParams1);
                DataCopyPad(inputTensor2, z[leftNums], copyInParams2, copyInPadParams2);
                inQueue.EnQue(inputTensor);
                inQueue.EnQue(inputTensor2);

                inputTensor = inQueue.DeQue<float>();
                inputTensor2 = inQueue.DeQue<float>();
                Mul(leftZ2, inputTensor, inputTensor, leftNums);
                Mul(rightZ2, inputTensor2, inputTensor2, rightNums);
                inQueue.FreeTensor(inputTensor);
                inQueue.FreeTensor(inputTensor2);
            }
            {
                // psi1(omega)
                inputTensor = inQueue.AllocTensor<float>();
                Adds<float>(leftOtherTmp, leftDelta, -omega, leftNums);
                Adds<float>(inputTensor, leftWork, omega, leftNums);
                Mul(leftOtherTmp, leftOtherTmp, inputTensor, leftNums); // d^2-omega^2
                Div(leftOtherTmp, leftZ2, leftOtherTmp, leftNums);
                ReduceSum(leftOtherTmp, leftOtherTmp, inputTensor, leftNums);
                inQueue.FreeTensor(inputTensor);
                psi1 = leftOtherTmp(0);
            }
            {
                // psi2(omega)
                inputTensor = inQueue.AllocTensor<float>();
                Adds<float>(rightOtherTmp, rightDelta, -omega, rightNums);
                Adds<float>(inputTensor, rightWork, omega, rightNums);
                Mul(rightOtherTmp, rightOtherTmp, inputTensor, rightNums); // d^2-omega^2
                Div(rightOtherTmp, rightZ2, rightOtherTmp, rightNums);
                ReduceSum(rightOtherTmp, rightOtherTmp, inputTensor, rightNums);
                inQueue.FreeTensor(inputTensor);
                psi2 = rightOtherTmp(0);
            }
            result = 1.0f + psi1 + psi2;
            if (result > 0)
            {
                // delta = dj-di,work=dj+di,dChosen=di
                dChosen = di;
            }
            else
            {
                // delta=dj-dip1,work=dj+dip1,dChosen=dip1
                dChosen = dip1;
            }
            // get final delta,work
            {
                Adds<float>(leftDelta, leftDelta, -dChosen, leftNums);
                Adds<float>(rightDelta, rightDelta, -dChosen, rightNums);
                Adds<float>(leftWork, leftWork, dChosen, leftNums);
                Adds<float>(rightWork, rightWork, dChosen, rightNums);
            }
            // get first miu
            miu = omega - dChosen;
            tol = 8.0f * (psi2 - psi1 + 1.0f) * n * EPS;
            uint16_t iter = 0;
            while (fabs(result) > tol && iter < MaxIter)
            {
                iter++;
                // calc psi1derivative psi2derivative,c1c2c3
                {
                    // psi1derivative
                    inputTensor = inQueue.AllocTensor<float>();
                    Adds<float>(leftOtherTmp, leftDelta, -miu, leftNums);
                    Adds<float>(inputTensor, leftWork, miu, leftNums);
                    Mul(leftOtherTmp, leftOtherTmp, inputTensor, leftNums); // d^2-omega^2
                    Mul(leftOtherTmp, leftOtherTmp, leftOtherTmp, leftNums);
                    Div(leftOtherTmp, leftZ2, leftOtherTmp, leftNums);
                    ReduceSum(leftOtherTmp, leftOtherTmp, inputTensor, leftNums);
                    inQueue.FreeTensor(inputTensor);
                    psi1derivative = leftOtherTmp(0);
                }
                {
                    // psi2derivative
                    inputTensor = inQueue.AllocTensor<float>();
                    Adds<float>(rightOtherTmp, rightDelta, -miu, rightNums);
                    Adds<float>(inputTensor, rightWork, miu, rightNums);
                    Mul(rightOtherTmp, rightOtherTmp, inputTensor, rightNums); // d^2-omega^2
                    Mul(rightOtherTmp, rightOtherTmp, rightOtherTmp, rightNums);
                    Div(rightOtherTmp, rightZ2, rightOtherTmp, rightNums);
                    ReduceSum(rightOtherTmp, rightOtherTmp, inputTensor, rightNums);
                    inQueue.FreeTensor(inputTensor);
                    psi2derivative = rightOtherTmp(0);
                }
                float coeff1 = (leftDelta(leftNums - 1) - miu) * (leftWork(leftNums - 1) + miu);
                c1 = psi1derivative * coeff1 * coeff1;
                c1hat = psi1 - psi1derivative * coeff1;
                coeff1 = (rightDelta(0) - miu) * (rightWork(0) + miu);
                c2 = psi2derivative * coeff1 * coeff1;
                c2hat = psi2 - psi2derivative * coeff1;
                c3 = 1 + c1hat + c2hat;

                // calc coeffs of quadratic equation
                float a = c3, negb = c1 + c2 + c3 * di2 + c3 * dip12, c = c1 * dip12 + c2 * di2 + c3 * di2 * dip12;
                float sigma1, sigma2;
                if (negb >= 0)
                {
                    sigma1 = sqrt((negb + sqrt(negb * negb - 4 * a * c)) / (2 * a));
                    sigma2 = sqrt(2 * c / (negb + sqrt(negb * negb - 4 * a * c)));
                }
                else
                {
                    sigma1 = sqrt(2 * c / (negb - sqrt(negb * negb - 4 * a * c)));
                    sigma2 = sqrt((negb - sqrt(negb * negb - 4 * a * c)) / (2 * a));
                }
                // get the new miu
                if (sigma1 > di && sigma1 < dip1)
                {
                    miu = sigma1 - dChosen;
                }
                else if (sigma2 > di && sigma2 < dip1)
                {
                    miu = sigma2 - dChosen;
                }
                else
                {
                    printf("panic:in SecularEquationSolver,sigma1:%f,sigma2:%f,di:%f,dip1:%f\n", sigma1, sigma2, di, dip1);
                    miu = sigma1 - dChosen;
                }

                // calculate new psi1,psi2,result
                {
                    // psi1(miu)
                    inputTensor = inQueue.AllocTensor<float>();
                    Adds<float>(leftOtherTmp, leftDelta, -miu, leftNums);
                    Adds<float>(inputTensor, leftWork, miu, leftNums);
                    Mul(leftOtherTmp, leftOtherTmp, inputTensor, leftNums); // d^2-omega^2
                    Div(leftOtherTmp, leftZ2, leftOtherTmp, leftNums);
                    ReduceSum(leftOtherTmp, leftOtherTmp, inputTensor, leftNums);
                    inQueue.FreeTensor(inputTensor);
                    psi1 = leftOtherTmp(0);
                }
                {
                    // psi2(miu)
                    inputTensor = inQueue.AllocTensor<float>();
                    Adds<float>(rightOtherTmp, rightDelta, -miu, rightNums);
                    Adds<float>(inputTensor, rightWork, miu, rightNums);
                    Mul(rightOtherTmp, rightOtherTmp, inputTensor, rightNums); // d^2-omega^2
                    Div(rightOtherTmp, rightZ2, rightOtherTmp, rightNums);
                    ReduceSum(rightOtherTmp, rightOtherTmp, inputTensor, rightNums);
                    inQueue.FreeTensor(inputTensor);
                    psi2 = rightOtherTmp(0);
                }
                result = 1.0f + psi1 + psi2;
                tol = 8.0f * (psi2 - psi1 + 1.0f) * n * EPS;
            }
            // update d and tmpSpace
            d(i) = dChosen + miu;
            // store d^2-sigma^2 in tmpSpace
            { // left
                inputTensor = inQueue.AllocTensor<float>();
                outputTensor = outQueue.AllocTensor<float>();
                Adds<float>(leftOtherTmp, leftDelta, -miu, leftNums);
                Adds<float>(inputTensor, leftWork, miu, leftNums);
                Mul(outputTensor, leftOtherTmp, inputTensor, leftNums); // d^2-omega^2
                inQueue.FreeTensor(inputTensor);
                outQueue.EnQue(outputTensor);

                outputTensor = outQueue.DeQue<float>();
                DataCopyPad(tmpSpace, outputTensor, copyOutParams1);
                outQueue.FreeTensor(outputTensor);
            }
            { // right
                inputTensor = inQueue.AllocTensor<float>();
                outputTensor = outQueue.AllocTensor<float>();
                Adds<float>(rightOtherTmp, rightDelta, -miu, rightNums);
                Adds<float>(inputTensor, rightWork, miu, rightNums);
                Mul(outputTensor, rightOtherTmp, inputTensor, rightNums); // d^2-omega^2
                inQueue.FreeTensor(inputTensor);
                outQueue.EnQue(outputTensor);
                outputTensor = outQueue.DeQue<float>();
                DataCopyPad(tmpSpace[leftNums], outputTensor, copyOutParams2);
                outQueue.FreeTensor(outputTensor);
            }
        }
        else
        {
            // i==n-1
            float znorm, psi, psiderivative, c1, c1hat, tol, result;
            const uint16_t UpperLen = (n + BlockFloatCnt - 1) / BlockFloatCnt * BlockFloatCnt;
            di = dsigma(i);
            di2 = di * di;
            LocalTensor<float> delta = tmpBuf1.Get<float>();
            LocalTensor<float> work = delta[UpperLen];
            LocalTensor<float> z2 = tmpBuf2.Get<float>();
            LocalTensor<float> otherTmp = z2[UpperLen];
            LocalTensor<float> inputTensor, outputTensor;
            const DataCopyExtParams copyInParams = {1, n * sizeOfFloat, 0, 0, 0};
            const DataCopyExtParams copyOutParams = {1, n * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyInPadParams = {true, 0, 0, 0.0f};
            { // get Delta and Work
                inputTensor = inQueue.AllocTensor<float>();
                DataCopyPad(inputTensor, dsigma, copyInParams, copyInPadParams);
                inQueue.EnQue(inputTensor);
                inputTensor = inQueue.DeQue<float>();
                Adds(delta, inputTensor, -di, n);
                Adds(work, inputTensor, di, n);
                inQueue.FreeTensor(inputTensor);

                inputTensor = inQueue.AllocTensor<float>();
                DataCopyPad(inputTensor, z, copyInParams, copyInPadParams);
                inQueue.EnQue(inputTensor);

                inputTensor = inQueue.DeQue<float>();
                Mul(z2, inputTensor, inputTensor, n);
                ReduceSum(otherTmp, z2, inputTensor, n);
                inQueue.FreeTensor(inputTensor);
                znorm = otherTmp(0);
            }
            dip1 = dsigma(i) + znorm;
            dip12 = dip1 * dip1;
            miu = znorm / 2.0f;
            // calc psi,result
            {
                inputTensor = inQueue.AllocTensor<float>();
                Adds(otherTmp, delta, -miu, n);
                Adds(inputTensor, work, miu, n);
                Mul(otherTmp, otherTmp, inputTensor, n); // d^2-omega^2
                Div(otherTmp, z2, otherTmp, n);
                ReduceSum(otherTmp, otherTmp, inputTensor, n);
                inQueue.FreeTensor(inputTensor);
                psi = otherTmp(0);
            }
            result = 1.0f + psi; // psi<0
            tol = 8.0f * (fabs(psi) + 1.0f) * n * EPS;
            // calc c1,c1hat
            uint16_t iter = 0;
            while (fabs(result - 1.0f) > tol && iter < MaxIter)
            {
                ++iter;
                {
                    // psiderivative
                    inputTensor = inQueue.AllocTensor<float>();
                    Adds<float>(otherTmp, delta, -miu, n);
                    Adds<float>(inputTensor, work, miu, n);
                    Mul(otherTmp, otherTmp, inputTensor, n); // d^2-omega^2
                    Mul(otherTmp, otherTmp, otherTmp, n);
                    Div(otherTmp, z2, otherTmp, n);
                    ReduceSum(otherTmp, otherTmp, inputTensor, n);
                    inQueue.FreeTensor(inputTensor);
                    psiderivative = otherTmp(0);
                }
                // calc c1,c1hat
                float coeff1 = (delta(n - 1) - miu) * (work(n - 1) + miu);
                c1 = psiderivative * coeff1 * coeff1;
                c1hat = psi - psiderivative * coeff1;
                miu = sqrt(di2 + c1 / (c1hat + 1.0f)) - di;

                // calc new psi,result
                {
                    inputTensor = inQueue.AllocTensor<float>();
                    Adds<float>(otherTmp, delta, -miu, n);
                    Adds<float>(inputTensor, work, miu, n);
                    Mul(otherTmp, otherTmp, inputTensor, n); // d^2-omega^2
                    Div(otherTmp, z2, otherTmp, n);
                    ReduceSum(otherTmp, otherTmp, inputTensor, n);
                    inQueue.FreeTensor(inputTensor);
                    psi = otherTmp(0);
                }
                result = 1.0f + psi;
                tol = 8.0f * (fabs(psi) + 1.0f) * n * EPS;
            }
            // update d and tmpSpace
            d(i) = di + miu;
            // store d^2-sigma^2 in tmpSpace
            {
                inputTensor = inQueue.AllocTensor<float>();
                outputTensor = outQueue.AllocTensor<float>();
                Adds<float>(otherTmp, delta, -miu, n);
                Adds<float>(inputTensor, work, miu, n);
                Mul(outputTensor, otherTmp, inputTensor, n); // d^2-omega^2
                inQueue.FreeTensor(inputTensor);
                outQueue.EnQue(outputTensor);
                outputTensor = outQueue.DeQue<float>();
                DataCopyPad(tmpSpace, outputTensor, copyOutParams);
                outQueue.FreeTensor(outputTensor);
            }
        }
        printf("[SecularEquationSolver] 计算完成, i=%d\n", i);
    }
    __aicore__ inline void MergeSubMatrix_step2(const uint16_t k, const uint16_t leftColNum, const uint16_t rightColNum, const bool isSquare, GlobalTensor<float> &leftSingularMatrix, GlobalTensor<float> &rightSingularMatrix, GlobalTensor<float> &d, GlobalTensor<float> &st, GlobalTensor<float> &gt, GlobalTensor<float> &f, GlobalTensor<float> &l, GlobalTensor<uint32_t> &idxq, GlobalTensor<uint32_t> &idxc, GlobalTensor<uint32_t> &ctot, GlobalTensor<float> &dsigma, GlobalTensor<float> &z, GlobalTensor<float> &tmpSpace)
    {
        printf("[MergeSubMatrix_step2] k=%d, leftColNum=%d, rightColNum=%d, isSquare=%d\n", k, leftColNum, rightColNum, isSquare);
        const auto leftRowNum = leftColNum - 1, rightRowNum = rightColNum - 1 + isSquare, totalRowNum = leftRowNum + rightRowNum + 1, totalColNum = leftColNum + rightColNum;
        LocalTensor<float> inputTensor, outputTensor;
        if (k == 1)
        {
            printf("[MergeSubMatrix_step2] 1x1 svd\n");
            // quick solve 1x1 svd
            d(0) = fabs(z(0));
            // copy st and gt to leftSingularMatrix and rightSingularMatrix
            copyRow(totalColNum, gt[0], rightSingularMatrix[0]);
            f(0) = gt(0);
            l(0) = gt(totalColNum - 1);
            bool isNeg = z(0) < 0;
            if (isNeg)
            {
                // st[0]*=-1搬到leftSingularMatrix[0]
                const DataCopyExtParams copyInParamsf = {1, totalRowNum * sizeOfFloat, 0, 0, 0};
                const DataCopyPadExtParams<float> copyInPadParamsf = {true, 0, 0, 0.0f};
                const DataCopyExtParams copyOutParamsf = {1, totalRowNum * sizeOfFloat, 0, 0, 0};
                inputTensor = inQueue.AllocTensor<float>();
                DataCopyPad(inputTensor, st[0], copyInParamsf, copyInPadParamsf);
                inQueue.EnQue(inputTensor);

                outputTensor = outQueue.AllocTensor<float>();
                inputTensor = inQueue.DeQue<float>();
                Muls<float>(outputTensor, inputTensor, -1, totalRowNum);
                outQueue.EnQue(outputTensor);
                inQueue.FreeTensor(inputTensor);

                outputTensor = outQueue.DeQue<float>();
                DataCopyPad(leftSingularMatrix[0], outputTensor, copyOutParamsf);
                outQueue.FreeTensor(outputTensor);
            }
            else
            {
                copyRow(totalRowNum, st[0], leftSingularMatrix[0]);
            }

            goto FormIdxq;
        }
        // first solve the singular values,roots of secular equation
        if (k == 2)
        {
            printf("[MergeSubMatrix_step2] 2x2 svd\n");
            float a11 = z(0), a12 = z(1), a22 = dsigma(1);
            // rank2,a11!=0,a12!=0,a22!=0
            float m11 = a11 * a11 + a12 * a12, m12 = a12 * a22, m22 = a22 * a22;
            float negb = m11 + m22;                         // negb >0
            float diff = m11 > m22 ? m11 - m22 : m22 - m11; // use bigger float to minus smaller float to get high precision
            float delta = diff * diff + 4 * m12 * m12;      // delta >0
            float sigma1 = sqrt((negb + sqrt(delta)) / 2.0f);
            float sigma2 = sqrt(2.0f * (a11 * a11 * a22 * a22) / (negb + sqrt(delta)));
            if (sigma1 > sigma2)
            {
                swap(sigma1, sigma2);
            }
            // ascending order
            d(0) = sigma1;
            d(1) = sigma2;
            float v1 = sigma1 * sigma1 - a12 * a12 - m22, v2 = a11 * a12, u1 = sigma1 * sigma1 - m22, u2 = m12;
            float normv = sqrt(v1 * v1 + v2 * v2), normu = sqrt(u1 * u1 + u2 * u2);
            rightSingularMatrix(0) = v1 / normv;
            rightSingularMatrix(1) = v2 / normv;
            leftSingularMatrix(0) = u1 / normu;
            leftSingularMatrix(1) = u2 / normu;
            v1 = sigma2 * sigma2 - a12 * a12 - m22, u1 = sigma2 * sigma2 - m22;
            normv = sqrt(v1 * v1 + v2 * v2), normu = sqrt(u1 * u1 + u2 * u2);
            rightSingularMatrix(LDN) = v1 / normv;
            rightSingularMatrix(LDN + 1) = v2 / normv;
            leftSingularMatrix(LDN) = u1 / normu;
            leftSingularMatrix(LDN + 1) = u2 / normu;
            RefreshAllCache();
            goto MatrixMul;
        }

        // k>=3,solve the secular equation
        // store dk^dk-sigmai^sigmai in tmpSpace(i,k)
        printf("[MergeSubMatrix_step2] k>=3,solve the secular equation\n");
        for (uint16_t i = 0; i < k; ++i)
        {
            SecularEquationSolver(k, i, dsigma, z, tmpSpace[i * LDN], d[i]);
        }
        RefreshAllCache(); // get d from cache
        // use lowner to compute z'
        float zi;
        for (uint16_t i = 0; i < k; ++i)
        {
            zi = tmpSpace((k - 1) * LDN + i); // di^2 -sigman^2
            for (uint16_t j = 0; j < i; ++j)
            {
                zi *= tmpSpace(j * LDN + i) / (dsigma(i) - dsigma(j)) / (dsigma(i) + dsigma(j));
            }
            for (uint16_t j = i; j < k - 1; ++j)
            {
                zi *= tmpSpace(j * LDN + i) / (dsigma(i) - dsigma(j + 1)) / (dsigma(i) + dsigma(j + 1));
            }
            z(i) = sign(sqrt(fabs(zi)), z(i));
        }

        // compute the left and right singular vectors
        // compute right singular vectors first
        {
            const DataCopyExtParams copyInParams = {1, k * sizeOfFloat, 0, 0, 0};
            const DataCopyExtParams copyOutParams = {1, k * sizeOfFloat, 0, 0, 0};
            const DataCopyPadExtParams<float> copyExtParams = {true, 0, 0, 0.0f};
            auto tmp = tmpBuf1.Get<float>();
            for (int i = 0; i < k; ++i)
            {
                // load d^2 -sigma i ^2
                inputTensor = inQueue.AllocTensor<float>();
                auto inputTensor2 = inQueue.AllocTensor<float>();
                DataCopyPad(inputTensor, tmpSpace[i * LDN], copyInParams, copyExtParams);
                DataCopyPad(inputTensor2, z, copyInParams, copyExtParams);
                inQueue.EnQue(inputTensor);
                inQueue.EnQue(inputTensor2);

                inputTensor = inQueue.DeQue<float>();
                inputTensor2 = inQueue.DeQue<float>();
                Div(tmp, inputTensor2, inputTensor, k); // z/(d^2-sigma^2)
                inQueue.FreeTensor(inputTensor);
                inQueue.FreeTensor(inputTensor2);

                inputTensor = inQueue.AllocTensor<float>();
                DataCopyPad(inputTensor, dsigma, copyInParams, copyExtParams); // d
                inQueue.EnQue(inputTensor);

                inputTensor = inQueue.DeQue<float>();
                outputTensor = outQueue.AllocTensor<float>();
                auto outputTensor2 = outQueue.AllocTensor<float>();
                ReduceSum(outputTensor2, tmp, outputTensor, k);
                float norm = outputTensor2(0);
                Muls(outputTensor2, tmp, 1.0f / norm, k);

                Mul(outputTensor, inputTensor, tmp, k);
                Duplicate(outputTensor, -1.0f, 1);
                ReduceSum(tmp, outputTensor, inputTensor, k);
                norm = tmp(0);
                Muls(outputTensor, outputTensor, 1.0f / norm, k);
                outQueue.EnQue(outputTensor);
                outQueue.EnQue(outputTensor2);
                inQueue.FreeTensor(inputTensor);

                outputTensor = outQueue.DeQue<float>();
                // 将outputTensor填充到leftSingularMatrix
                DataCopyPad(leftSingularMatrix[i * LDN], outputTensor, copyOutParams);
                outQueue.FreeTensor(outputTensor);

                // 取出outputTensor2并填充到rightSingularMatrix
                outputTensor2 = outQueue.DeQue<float>();
                DataCopyPad(rightSingularMatrix[i * LDN], outputTensor2, copyOutParams);
                outQueue.FreeTensor(outputTensor2);
            }
        }

    MatrixMul:
        // multiply with prior orthonormal matrix to get the final left and right singular vectors
        // prior matrix stored in st and gt,new matrix stored in leftSingularMatrix and rightSingularMatrix,use tmpSpace as dst,then copy back
        // update the l and f as well
        // left Matrix = st*leftMatrix ,st stored in leftSingularMatrix and leftMatrix stored in st.It's somewhat perplexing,though
        printf("[MergeSubMatrix_step2] multiply with prior orthonormal matrix to get the final left and right singular vectors\n");
        mm->SetOrgShape(LDN, LDN, LDN, LDN);
        mm->SetSingleShape(k, k, totalRowNum);
        mm->SetTensorA(leftSingularMatrix);
        mm->SetTensorB(st);
        mm->IterateAll(tmpSpace);
        CopyMatrix(tmpSpace, leftSingularMatrix, LDN, LDN, k, totalRowNum);

        mm->SetOrgShape(LDN, LDN, LDN, LDN);
        mm->SetSingleShape(k, k, totalColNum);
        mm->SetTensorA(rightSingularMatrix);
        mm->SetTensorB(gt);
        mm->IterateAll(tmpSpace);
        CopyMatrix(tmpSpace, rightSingularMatrix, LDN, LDN, k, totalColNum);
        for (int i = 0; i < k; ++i)
        {
            f(i) = rightSingularMatrix(i * LDN);
            l(i) = rightSingularMatrix(i * LDN + k - 1);
        }

    FormIdxq:
        RefreshAllCache(); // get d from cache
        printf("[MergeSubMatrix_step2] sort new d to form idxq\n");
        // sort new d to form idxq
        {
            // {
            //     //init new idxq for MrgSort
            //     auto outputTensor = outQueue.AllocTensor<uint32_t>();
            //     CreateVecIndex<int32_t>(outputTensor.ReinterpretCast<int32_t>(), 0, totalRowNum);
            //     outQueue.EnQue(outputTensor);
            //     outputTensor = outQueue.DeQue<uint32_t>();
            //     DataCopyPad(idxq, outputTensor, {1, totalRowNum * sizeOfUint32_t, 0, 0, 0});
            //     outQueue.FreeTensor(outputTensor);
            // }
            // construct MrgSort list.4B score 4B index.8n,8n+4
            {
                constructMrgSortList(tmpBuf1.Get<float>(), d, totalRowNum);
            }
            printf("[MergeSubMatrix_step2] MrgSort the mrglist,k*2=%d\n", k * 2);
            // MrgSort the mrglist
            {
                LocalTensor<float> workTensor = tmpBuf1.Get<float>();
                LocalTensor<float> dstTensor = tmpBuf2.Get<float>();
                MrgSortSrcList<float> mrgSortSrcList;
                mrgSortSrcList.src1 = workTensor;
                mrgSortSrcList.src2 = workTensor[k * 2];
                printf("[MergeSubMatrix_step2] mrgSortSrcList.src1=%p,mrgSortSrcList.src2=%p\n", mrgSortSrcList.src1.GetPhyAddr(), mrgSortSrcList.src2.GetPhyAddr());
                MrgSort4Info params;
                params.elementLengths[0] = k;
                params.elementLengths[1] = totalRowNum - k;
                params.ifExhaustedSuspension = false;
                params.validBit = 3;
                params.repeatTimes = 1;
                singleDumpTensor(workTensor, 64);
                MrgSort(dstTensor, mrgSortSrcList, params);
            }
            printf("[MergeSubMatrix_step2] get the idxq\n");
            // get the idxq
            {
                const DataCopyExtParams copyOutParamsi = {1, totalRowNum * sizeOfUint32_t, 0, 0, 0};

                LocalTensor<uint32_t> mrglisti = tmpBuf2.Get<uint32_t>();

                auto indexTensor = inQueue.AllocTensor<uint32_t>();
                auto outIndexTensor = outQueue.AllocTensor<uint32_t>();
                CreateVecIndex<int32_t>(indexTensor.ReinterpretCast<int32_t>(), 0, totalRowNum);
                Muls<int32_t>(indexTensor.ReinterpretCast<int32_t>(), indexTensor.ReinterpretCast<int32_t>(), 8, totalRowNum);
                Gather(outIndexTensor, mrglisti, indexTensor, 4, totalRowNum);
                outQueue.EnQue(outIndexTensor);
                inQueue.FreeTensor(indexTensor);

                outIndexTensor = outQueue.DeQue<uint32_t>();
                DataCopyPad(idxq[0], outIndexTensor, copyOutParamsi);
                outQueue.FreeTensor(outIndexTensor);
            }
        }

        printf("[MergeSubMatrix_step2] 结束\n");
        return;
    }

    __aicore__ inline void compute_base_case_svd(const SVDSubmatrixInfo &subMatrix)
    {
        const auto idx_start = subMatrix.start_col;
        const auto colNum = subMatrix.end_col - idx_start;
        const auto rowNum = subMatrix.end_col == LDN ? colNum : colNum - 1;
        printf("compute_base_case_svd,start_col:%d,colNum:%d,rowNum:%d\n", idx_start, colNum, rowNum);
        GlobalTensor<float> qt = qtGm[idx_start * LDN + idx_start];
        GlobalTensor<float> wt = wtGm[idx_start * LDN + idx_start];
        GlobalTensor<float> d = dGm[idx_start];
        GlobalTensor<uint32_t> idxq = idxqGm[idx_start];
        GlobalTensor<float> f = fGm[idx_start];
        GlobalTensor<float> l = lGm[idx_start];
        // allow e to be valid in case 1x1
        GlobalTensor<float> e = idx_start == LDN - 1 ? dGm[idx_start] : eGm[idx_start];
        if (colNum == 3)
        {
            compute_2x3_svd(qt, wt, d, e, idxq, f, l);
        }
        else if (colNum == 2 && rowNum == 2)
        {
            compute_2x2_svd(qt, wt, d, e, idxq, f, l);
        }
        else if (colNum == 2 && rowNum == 1)
        {
            compute_1x2_svd(qt, wt, d, e, idxq, f, l);
        }
        else if (colNum == 1 && rowNum == 1)
        {
            compute_1x1_svd(qt, wt, d, idxq, f, l);
        }
    }
    __aicore__ inline void compute_2x3_svd(GlobalTensor<float> &qt, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<float> &e, GlobalTensor<uint32_t> &idxq, GlobalTensor<float> &f, GlobalTensor<float> &l)
    {
        float a11 = d(0), a12 = e(0), a22 = d(1), a23 = e(1);
        idxq(0) = 1;
        idxq(1) = 0;
        if (a11 == 0 && a12 == 0 && a22 == 0 && a23 == 0)
        {
            // rank0
            d(0) = 0;
            d(1) = 0;
        }
        else if (a11 == 0 && a12 == 0)
        {
            // rank1 第一行为0
            float sq = sqrt(a22 * a22 + a23 * a23);
            d(0) = sq;
            d(1) = 0;
            qt(0) = 0;
            qt(1) = 1;
            qt(LDN) = 1;
            qt(LDN + 1) = 0;
            wt(0) = 0;
            wt(1) = a22 / sq;
            wt(2) = a23 / sq;
            wt(LDN) = 1;
            wt(LDN + 1) = 0;
            // wt(LDN + 2) = 0;
            // wt(2 * LDN) = 0;
            wt(2 * LDN + 1) = a23 / sq;
            wt(2 * LDN + 2) = -a22 / sq;
        }
        else if (a22 == 0 && a23 == 0)
        {
            // rank1 第二行为0
            float sq = sqrt(a11 * a11 + a12 * a12);
            d(0) = sq;
            d(1) = 0;
            // qt is unit
            wt(0) = a11 / sq;
            wt(1) = a12 / sq;
            wt(LDN) = a12 / sq;
            wt(LDN + 1) = -a11 / sq;
        }
        else if (a11 == 0 && a23 == 0)
        {
            // rank1, 第二列非0向量
            float sq = sqrt(a12 * a12 + a22 * a22);
            d(0) = sq;
            d(1) = 0;
            qt(0) = a12 / sq;
            qt(1) = a22 / sq;
            qt(LDN) = a22 / sq;
            qt(LDN + 1) = -a12 / sq;
            wt(0) = 0;
            wt(1) = 1;
            wt(LDN) = 1;
            wt(LDN + 1) = 0;
        }
        else
        {
            // rank2，compute AAt's eigenvalue
            float m11 = a11 * a11 + a12 * a12, m12 = a12 * a22, m22 = a22 * a22 + a23 * a23;
            float negb = m11 + m22;                         // negb >0
            float diff = m11 > m22 ? m11 - m22 : m22 - m11; // use bigger float to minus smaller float to get high precision
            float delta = diff * diff + 4 * m12 * m12;      // delta >0
            float sigma1 = sqrt((negb + sqrt(delta)) / 2.0f);
            float sigma2 = sqrt(2.0f * (m11 * m22 - m12 * m12) / (negb + sqrt(delta)));
            if (sigma1 < sigma2)
            {
                swap(sigma1, sigma2);
            }
            d(0) = sigma1;
            d(1) = sigma2;
            float u1 = sigma1 * sigma1 - m22, u2 = m12, v1 = a11 * u1, v2 = a12 * (sigma1 * sigma1 - a23 * a23), v3 = a12 * a22 * a23;
            float normu = sqrt(u1 * u1 + u2 * u2), normv = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
            qt(0) = u1 / normu;
            qt(1) = u2 / normu;
            wt(0) = v1 / normv;
            wt(1) = v2 / normv;
            wt(2) = v3 / normv;
            u1 = sigma2 * sigma2 - m22, v1 = a11 * u1, v2 = a12 * (sigma2 * sigma2 - a23 * a23);
            normu = sqrt(u1 * u1 + u2 * u2), normv = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
            qt(LDN) = u1 / normu;
            qt(LDN + 1) = u2 / normu;
            wt(LDN) = v1 / normv;
            wt(LDN + 1) = v2 / normv;
            wt(LDN + 2) = v3 / normv;
            v1 = a12 * a23, v2 = -a11 * a23, v3 = a11 * a22;
            normv = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
            wt(2 * LDN) = v1 / normv;
            wt(2 * LDN + 1) = v2 / normv;
            wt(2 * LDN + 2) = v3 / normv;
        }
        f(0) = wt(0);
        f(1) = wt(LDN);
        f(2) = wt(2 * LDN);
        l(0) = wt(2);
        l(1) = wt(2 + LDN);
        l(2) = wt(2 * LDN + 2);
    }
    __aicore__ inline void compute_2x2_svd(GlobalTensor<float> &qt, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<float> &e, GlobalTensor<uint32_t> &idxq, GlobalTensor<float> &f, GlobalTensor<float> &l)
    {
        float a11 = d(0), a12 = e(0), a22 = d(1);

        idxq(0) = 1;
        idxq(1) = 0;
        if (a11 == 0 && a12 == 0 && a22 == 0)
        {
            // rank 0
            d(0) = 0;
            d(1) = 0;
            // qt(0) = 1.0f;
            // qt(LDN + 1) = 1.0f;
            // wt(0) = 1.0f;
            // wt(LDN + 1) = 1.0f;
        }
        else if (a22 == 0)
        {
            // rank 1,a22==0
            float sq = sqrt(a11 * a11 + a12 * a12);
            d(0) = sq;
            d(1) = 0;
            // qt(0) = 1.0f;
            // qt(LDN + 1) = 1.0f;
            wt(0) = a11 / sq;
            wt(1) = a12 / sq;
            wt(LDN) = a12 / sq;
            wt(LDN + 1) = -a11 / sq;
        }
        else if (a11 == 0)
        {
            // rank 1,a11==0
            float sq = sqrt(a12 * a12 + a22 * a22);
            d(0) = sq;
            d(1) = 0;
            qt(0) = a12 / sq;
            qt(1) = a22 / sq;
            qt(LDN) = a22 / sq;
            qt(LDN + 1) = -a12 / sq;
            wt(0) = 0;
            wt(1) = 1;
            wt(LDN) = 1;
            wt(LDN + 1) = 0;
        }
        else if (a12 == 0)
        {
            // diagonal
            if (fabs(a11) >= fabs(a22))
            {
                d(0) = fabs(a11);
                d(1) = fabs(a22);
                qt(0) = sign(1.0f, a11);
                qt(LDN + 1) = sign(1.0f, a22);
            }
            else
            {
                d(0) = fabs(a22);
                d(1) = fabs(a11);
                qt(0) = 0.0f;
                qt(1) = sign(1.0f, a22);
                qt(LDN) = sign(1.0f, a11);
                qt(LDN + 1) = 0.0f;
                wt(0) = 0.0f;
                wt(1) = 1.0f;
                wt(LDN) = 1.0f;
                wt(LDN + 1) = 0.0f;
            }
        }
        else
        {
            // rank2,a11!=0,a12!=0,a22!=0
            float m11 = a11 * a11 + a12 * a12, m12 = a12 * a22, m22 = a22 * a22;
            float negb = m11 + m22;                         // negb >0
            float diff = m11 > m22 ? m11 - m22 : m22 - m11; // use bigger float to minus smaller float to get high precision
            float delta = diff * diff + 4 * m12 * m12;      // delta >0
            float sigma1 = sqrt((negb + sqrt(delta)) / 2.0f);
            float sigma2 = sqrt(2.0f * (a11 * a11 * a22 * a22) / (negb + sqrt(delta)));
            if (sigma1 < sigma2)
            {
                swap(sigma1, sigma2);
            }
            d(0) = sigma1;
            d(1) = sigma2;
            float v1 = sigma1 * sigma1 - a12 * a12 - m22, v2 = a11 * a12, u1 = sigma1 * sigma1 - m22, u2 = m12;
            float normv = sqrt(v1 * v1 + v2 * v2), normu = sqrt(u1 * u1 + u2 * u2);
            wt(0) = v1 / normv;
            wt(1) = v2 / normv;
            qt(0) = u1 / normu;
            qt(1) = u2 / normu;
            v1 = sigma2 * sigma2 - a12 * a12 - m22, u1 = sigma2 * sigma2 - m22;
            normv = sqrt(v1 * v1 + v2 * v2), normu = sqrt(u1 * u1 + u2 * u2);
            wt(LDN) = v1 / normv;
            wt(LDN + 1) = v2 / normv;
            qt(LDN) = u1 / normu;
            qt(LDN + 1) = u2 / normu;
        }
        f(0) = wt(0);
        f(1) = wt(LDN);
        l(0) = wt(1);
        l(1) = wt(1 + LDN);
    }
    __aicore__ inline void compute_1x2_svd(GlobalTensor<float> &qt, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<float> &e, GlobalTensor<uint32_t> &idxq, GlobalTensor<float> &f, GlobalTensor<float> &l)
    {
        float a11 = d(0), a12 = e(0);
        float sq = sqrt(a11 * a11 + a12 * a12);
        idxq(0) = 0;
        d(0) = sq;
        // qt(0) = 1.0f;
        if (sq == 0)
        {
            // wt(0) = 1.0f;
            // wt(1) = 0;
            // wt(LDN) = 0;
            // wt(LDN + 1) = 1.0f;
        }
        else
        {
            wt(0) = a11 / sq;
            wt(1) = a12 / sq;
            wt(LDN) = a12 / sq;
            wt(LDN + 1) = -a11 / sq;
        }
        f(0) = wt(0);
        f(1) = wt(LDN);
        l(0) = wt(1);
        l(1) = wt(1 + LDN);
    }
    __aicore__ inline void compute_1x1_svd(GlobalTensor<float> &qt, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<uint32_t> &idxq, GlobalTensor<float> &f, GlobalTensor<float> &l)
    {
        idxq(0) = 0;
        float a11 = d(0);
        qt(0) = sign(1.0f, a11);
        d(0) = fabs(a11);
        // wt(0) = 1.0f;
        // no need to set w to 1.0f because it initializes to unit matrix
        f(0) = wt(0);
        l(0) = wt(0);
    }

    __aicore__ inline void rearrange_qwtAccordingToIdxq()
    {
    }

    __aicore__ inline void updateUVt()
    {
        NotParallelQuiter;
        // update UVt from Q Wt
        // LDN columns of U  are updated
        // all rows of Vt are updated
        RefreshAllCache();
        printf("before updateUVt\n");
        // singleDumpTensor(uGm, 1024);
        // singleDumpTensor(vtGm, 1024);
        // singleDumpTensor(qtGm, 1024);
        // singleDumpTensor(wtGm, 1024);

        mm->SetOrgShape(LDM, LDN, LDM, LDN);
        // printf("after setOrgShape\n");
        mm->SetSingleShape(LDM, LDN, LDN);
        // printf("after setSingleShape\n");
        mm->SetTensorA(uGm);
        // printf("after setTensorA\n");
        mm->SetTensorB(qtGm, true);
        // printf("after setTensorB\n");
        mm->IterateAll(tmpGm);
        // printf("after iterateAll\n");
        // printf("%f %f %f %f\n", tmpGm(0), tmpGm(1), tmpGm(2), tmpGm(3));

        // mm->End();
        // printf("after end\n");
        // Copy tmpGm to uGm
        printf("before cache refresh\n");
        // singleDumpTensor(tmpGm, 1024);
        // singleDumpTensor(uGm, 1024);
        // singleDumpTensor(wtGm, 1024);
        // singleDumpTensor(vtGm, 1024);

        // AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(tmpGm);
        // printf("before copyMatrix\n");
        // singleDumpTensor(tmpGm, 1024);
        // singleDumpTensor(uGm, 1024);
        // singleDumpTensor(wtGm, 1024);
        // singleDumpTensor(vtGm, 1024);

        CopyMatrix(tmpGm, uGm, LDN, LDM, LDM, LDN);

        // printf("after copyMatrix\n");
        // singleDumpTensor(uGm, 1024);
        // printf("after updateU\n");
        // printf("before updateVt\n");
        // singleDumpTensor(wtGm, 1024);
        // singleDumpTensor(vtGm, 1024);

        mm->SetOrgShape(LDN, LDN, LDN, LDN);
        mm->SetSingleShape(LDN, LDN, LDN);
        mm->SetTensorA(wtGm);
        mm->SetTensorB(vtGm);
        mm->IterateAll(tmpGm);
        mm->End();
        // printf("after end\n");
        // printf("before DataCacheCleanAndInvalid\n");
        // singleDumpTensor(tmpGm, 1024);

        // AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(uGm);
        // Copy tmpGm to vtGm
        // printf("before copyMatrix\n");
        // singleDumpTensor(tmpGm, 1024);
        CopyMatrix(tmpGm, vtGm, LDN, LDN, LDN, LDN);
        // printf("after updateVt\n");
        // singleDumpTensor(vtGm, 1024);

        return;
    }
    __aicore__ inline void CopyMatrix(GlobalTensor<float> &src, GlobalTensor<float> &dst, uint16_t src_n, uint16_t dst_n, uint16_t copyM, uint16_t copyN)
    {
        NotParallelQuiter;
        DataCopyExtParams copyInParams = {1, copyN * sizeOfFloat, 0, 0, 0};
        DataCopyExtParams copyOutParams = {1, copyN * sizeOfFloat, 0, 0, 0};
        DataCopyPadExtParams<float> copyInPadParams = {true, 0, 0, 0.0f};

        for (int i = 0; i < copyM; i++)
        {
            auto bindLocalf = copyBind.AllocTensor<float>();
            DataCopyPad(bindLocalf, src[i * src_n], copyInParams, copyInPadParams);
            copyBind.EnQue(bindLocalf);

            bindLocalf = copyBind.DeQue<float>();
            DataCopyPad(dst[i * dst_n], bindLocalf, copyOutParams);
            copyBind.FreeTensor(bindLocalf);
        }
    }

    __aicore__ inline void copyRow(uint16_t rowLen, GlobalTensor<float> src, GlobalTensor<float> dst)
    {
        DataCopyExtParams copyInParams = {1, rowLen * sizeOfFloat, 0, 0, 0};
        DataCopyExtParams copyOutParams = {1, rowLen * sizeOfFloat, 0, 0, 0};
        DataCopyPadExtParams<float> copyInPadParams = {true, 0, 0, 0.0f};

        auto bindLocalf = copyBind.AllocTensor<float>();
        DataCopyPad(bindLocalf, src, copyInParams, copyInPadParams);
        copyBind.EnQue(bindLocalf);

        bindLocalf = copyBind.DeQue<float>();
        DataCopyPad(dst, bindLocalf, copyOutParams);
        copyBind.FreeTensor(bindLocalf);
    }
    __aicore__ inline SVDSubmatrixInfo getSVDSubmatrixInfo(uint16_t idx)
    {
        return {svdStackGm(2 * idx), svdStackGm(2 * idx + 1)};
    }
    __aicore__ inline void setSVDSubmatrixInfo(uint16_t idx, SVDSubmatrixInfo info)
    {
        svdStackGm(2 * idx) = info.start_col;
        svdStackGm(2 * idx + 1) = info.end_col;
    }
    // x'=cx+sy, y'=-sx+cy
    __aicore__ inline void rotateRow(uint16_t rowLen, GlobalTensor<float> x, GlobalTensor<float> y, float c, float s)
    {
        const DataCopyExtParams copyInParams = {1, rowLen * sizeOfFloat, 0, 0, 0};
        const DataCopyExtParams copyOutParams = {1, rowLen * sizeOfFloat, 0, 0, 0};
        const DataCopyPadExtParams<float> copyInPadParams = {true, 0, 0, 0.0f};
        LocalTensor<float> xin, yin, xout, yout, tmp1, tmp2;
        tmp1 = tmpBuf1.Get<float>();
        tmp2 = tmpBuf2.Get<float>();
        xin = inQueue.AllocTensor<float>();
        DataCopyPad(xin, x, copyInParams, copyInPadParams);
        yin = inQueue.AllocTensor<float>();
        DataCopyPad(yin, y, copyInParams, copyInPadParams);
        inQueue.EnQue(xin);
        inQueue.EnQue(yin);

        xin = inQueue.DeQue<float>();
        yin = inQueue.DeQue<float>();
        xout = outQueue.AllocTensor<float>();
        yout = outQueue.AllocTensor<float>();
        Muls(tmp1, xin, c, rowLen);
        Muls(tmp2, yin, s, rowLen);
        Add(xout, tmp1, tmp2, rowLen);
        Muls(tmp1, xin, -s, rowLen);
        Muls(tmp2, yin, c, rowLen);
        Add(yout, tmp1, tmp2, rowLen);
        outQueue.EnQue(xout);
        outQueue.EnQue(yout);
        inQueue.FreeTensor(xin);
        inQueue.FreeTensor(yin);

        xout = outQueue.DeQue<float>();
        yout = outQueue.DeQue<float>();
        DataCopyPad(x, xout, copyOutParams);
        DataCopyPad(y, yout, copyOutParams);
        outQueue.FreeTensor(xout);
        outQueue.FreeTensor(yout);
    }
    __aicore__ inline void constructMrgSortList(const LocalTensor<float> &dst, const GlobalTensor<float> &src, const uint32_t count)
    {
        LocalTensor<uint32_t> dsti = dst.ReinterpretCast<uint32_t>();
        for (int i = 0; i < count; ++i)
        {
            dst(2 * i) = -src(i); // to make it descending order
            dsti(2 * i + 1) = i;
        }
        RefreshAllCache();
    }

    __aicore__ inline void RefreshAllCache()
    {
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(uGm);
    }

private:
    uint16_t LDM, LDN;
    // in the beginning u and vt are orthonormal matrix, generated by bidiagonalization
    // qt and wt are orthonormal matrix, generated by BDC
    // altogether,the singular matrix are uq,wtvt which
    // would in the end be stored in uGm and vtGm
    // d would be used to store singular values,and e would store the intermediate z;
    GlobalTensor<float> tmpGm, uGm, vtGm, dGm, eGm, qtGm, wtGm;
    GlobalTensor<float> stGm, gtGm, fGm, lGm;
    GlobalTensor<uint32_t> idxqGm;
    GlobalTensor<float> sigmaGm, zGm, dsigmaGm;
    GlobalTensor<uint32_t> idxpGm;
    GlobalTensor<uint32_t> idxGm;
    GlobalTensor<uint32_t> idxcGm;
    GlobalTensor<uint32_t> coltypGm;

    GlobalTensor<uint16_t> svdStackGm;
    BDCMatmulType *mm;
    const uint8_t blockIdx, blockNum;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> tmpBuf1, tmpBuf2;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> copyBind;
    // TBuf<TPosition::VECCALC> svdWorkspaceBuf;
    SVDTiling *svdTiling;
};

extern "C" __global__ __aicore__ void svd_DC(int M, int N, GM_ADDR a, GM_ADDR u, GM_ADDR vt, GM_ADDR d, GM_ADDR e, GM_ADDR qt, GM_ADDR wt, GM_ADDR idx, GM_ADDR workspace, GM_ADDR tilingGM)
{
    TPipe pipe;
    SVDTiling tiling;
    GM_ADDR svdStack;
    CopyTiling(&tiling, &svdStack, tilingGM);
    BDCMatmulType mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling.matmultiling); // Initialize the matmul object.
#ifdef __DAV_C220_VEC__
    BDC bdc;
    bdc.init(M, N, a, u, vt, d, e, qt, wt, idx, svdStack, workspace, &tiling, pipe, mm);
    bdc.Process();
#endif
    // for (int len = 1; len <= length; len *= 2)
    // {
    //     for (int l1 = st; l1 <= ed; l1 += 2 * len)
    //     {
    //         int r1 = l1 + len - 1;
    //         int l2 = l1 + len;
    //         int r2 = l1 + 2 * len - 1;
    //         if (r1 >= ed)
    //         {
    //             continue;
    //         }
    //         if (r2 >= ed)
    //         {
    //             r2 = ed;
    //         }
    //         printf("INNER_MERGING: (%d,%d),(%d,%d)\n", l1, r1, l2, r2);
    //         // merge(l1,r1,l2,r2);
    //     }
    // }
    // for (int len = 1; len <= num; len *= 2)
    // {
    //     printf("outter len:%d\n", len);
    //     if (idx % (2 * len))
    //     {
    //         // CrossCoreSetFlag<0x0, PIPE_S>(0x8);
    //         // CrossCoreWaitFlag(0x8);
    //         continue;
    //     }
    //     int l1 = idx;
    //     int r1 = idx + len - 1;
    //     int l2 = idx + len;
    //     int r2 = idx + 2 * len - 1;
    //     if (r1 >= num - 1)
    //     {
    //         // CrossCoreSetFlag<0x0, PIPE_S>(0x8);
    //         // CrossCoreWaitFlag(0x8);
    //         continue;
    //     }
    //     if (r2 >= num - 1)
    //     {
    //         r2 = num - 1;
    //     }
    //     printf("OUTTER_MERGING: (%d,%d),(%d,%d)\n", l1, r1, l2, r2);
    //     // CrossCoreSetFlag<0x0, PIPE_S>(0x8);
    //     // CrossCoreWaitFlag(0x8);
    // }
}
