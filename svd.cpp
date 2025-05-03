#include "kernel_operator.h"
#include <lib/matmul_intf.h>

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
constexpr float EPS = 1e-6;
constexpr uint32_t sizeOfFloat = sizeof(float);
constexpr int32_t BlockSize = 32;
constexpr int32_t BlockFloatCnt = BlockSize / sizeOfFloat;
constexpr int32_t SizePerOperation = 256;
constexpr int32_t BlockNumPerOperation = SizePerOperation / BlockSize;
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

        for (uint32_t i = 0; i < sizeof(SVDTiling) / sizeof(uint32_t); i++, ptr++)
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
    __aicore__ inline void init(int M, int N, GM_ADDR a, GM_ADDR u, GM_ADDR vt, GM_ADDR d, GM_ADDR e, GM_ADDR qt, GM_ADDR wt, GM_ADDR idx, GM_ADDR svdStack, GM_ADDR workspace, SVDTiling *tiling, TPipe &pipe,BDCMatmulType &inputmm)
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
        idxqGm.SetGlobalBuffer((__gm__ uint32_t *)idx, N);
        pipe.InitBuffer(copyBind, BUFFER_NUM, N * sizeOfFloat + 32);

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
                compute_2x2_svd(qtGm, wtGm, dGm, eGm, idxqGm);
            }
            updateUVt();

            return;
        }
        else if (LDN == 1)
        {
            if (blockIdx == 0)
            {
                compute_1x1_svd(qtGm, wtGm, dGm, idxqGm);
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
        // acquire singular matrix of subproblems
        // form d and z
        // rotate to remove the N+1 column if necessary
        // get idxq
        // sort d and get another permutation
        // deflate z1
        // deflate z and d
        // permute qt and wt
        // call secular quation solver to get sigma and singular vectors
        // update singular vectors with matmul
        // sort sigma and form idxq
        return;
    }

    __aicore__ inline void compute_base_case_svd(const SVDSubmatrixInfo &subMatrix)
    {
        const auto idx_start = subMatrix.start_col;
        const auto colNum = subMatrix.end_col - idx_start + 1;
        const auto rowNum = subMatrix.end_col == LDN ? colNum : colNum - 1;
        GlobalTensor<float> qt = qtGm[idx_start * LDN + idx_start];
        GlobalTensor<float> wt = wtGm[idx_start * LDN + idx_start];
        GlobalTensor<float> d = dGm[idx_start];
        GlobalTensor<uint32_t> idxq = idxqGm[idx_start];
        // allow e to be valid in case 1x1
        GlobalTensor<float> e = idx_start == LDN - 1 ? dGm[idx_start] : eGm[idx_start];
        if (colNum == 3)
        {
            compute_2x3_svd(qt, wt, d, e, idxq);
        }
        else if (colNum == 2 && rowNum == 2)
        {
            compute_2x2_svd(qt, wt, d, e, idxq);
        }
        else if (colNum == 2 && rowNum == 1)
        {
            compute_1x2_svd(qt, wt, d, e, idxq);
        }
        else if (colNum == 1 && rowNum == 1)
        {
            compute_1x1_svd(qt, wt, d, idxq);
        }
    }
    __aicore__ inline void compute_2x3_svd(GlobalTensor<float> &qt, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<float> &e, GlobalTensor<uint32_t> &idxq)
    {
        float a11 = d(0), a12 = e(0), a22 = d(1), a23 = e(1);
        idxq(0) = 0;
        idxq(1) = 1;
        if (a11 == 0 && a12 == 0 && a22 == 0 && a23 == 0)
        {
            // rank0
            d(0) = 0;
            d(1) = 0;
            return;
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
            return;
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

            return;
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
            return;
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
    }
    __aicore__ inline void compute_2x2_svd(GlobalTensor<float> &qt, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<float> &e, GlobalTensor<uint32_t> &idxq)
    {
        float a11 = d(0), a12 = e(0), a22 = d(1);

        idxq(0) = 0;
        idxq(1) = 1;
        if (a11 == 0 && a12 == 0 && a22 == 0)
        {
            // rank 0
            d(0) = 0;
            d(1) = 0;
            // qt(0) = 1.0f;
            // qt(LDN + 1) = 1.0f;
            // wt(0) = 1.0f;
            // wt(LDN + 1) = 1.0f;
            return;
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
            return;
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
            return;
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
            return;
        }
    }
    __aicore__ inline void compute_1x2_svd(GlobalTensor<float> &qt, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<float> &e, GlobalTensor<uint32_t> &idxq)
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
            return;
        }
        else
        {
            wt(0) = a11 / sq;
            wt(1) = a12 / sq;
            wt(LDN) = a12 / sq;
            wt(LDN + 1) = -a11 / sq;
            return;
        }
    }
    __aicore__ inline void compute_1x1_svd(GlobalTensor<float> &qt, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<uint32_t> &idxq)
    {
        idxq(0) = 0;
        float a11 = d(0);
        qt(0) = sign(1.0f, a11);
        d(0) = fabs(a11);
        // wt(0) = 1.0f;
        // no need to set w to 1.0f because it initializes to unit matrix
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
        AscendC ::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(uGm);
        printf("before updateUVt\n");
        singleDumpTensor(uGm, 16);
        singleDumpTensor(vtGm, 16);
        singleDumpTensor(qtGm, 16);
        singleDumpTensor(wtGm, 16);

        mm->SetOrgShape(LDM, LDN, LDM, LDN);
        printf("after setOrgShape\n");
        mm->SetSingleShape(LDM, LDN, LDN);
        printf("after setSingleShape\n");
        mm->SetTensorA(uGm);
        printf("after setTensorA\n");
        mm->SetTensorB(qtGm, true);
        printf("after setTensorB\n");
        mm->IterateAll(tmpGm);
        printf("after iterateAll\n");
        // printf("%f %f %f %f\n", tmpGm(0), tmpGm(1), tmpGm(2), tmpGm(3));

        // mm->End();
        // printf("after end\n");
        // Copy tmpGm to uGm
        printf("before cache refresh\n");
        singleDumpTensor(tmpGm, 16);
        singleDumpTensor(uGm, 16);
        singleDumpTensor(wtGm, 16);
        singleDumpTensor(vtGm, 16);

        // AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(tmpGm);
        printf("before copyMatrix\n");
        singleDumpTensor(tmpGm, 16);
        singleDumpTensor(uGm, 16);
        singleDumpTensor(wtGm, 16);
        singleDumpTensor(vtGm, 16);

        CopyMatrix(tmpGm, uGm, LDN, LDM, LDM, LDN);

        printf("after copyMatrix\n");
        singleDumpTensor(uGm, 16);
        printf("after updateU\n");
        printf("before updateVt\n");
        singleDumpTensor(wtGm, 16);
        singleDumpTensor(vtGm, 16);

        mm->SetOrgShape(LDN, LDN, LDN, LDN);
        mm->SetSingleShape(LDN, LDN, LDN);
        mm->SetTensorA(wtGm);
        mm->SetTensorB(vtGm);
        mm->IterateAll(tmpGm);
        mm->End();
        printf("after end\n");
        printf("before DataCacheCleanAndInvalid\n");
        singleDumpTensor(tmpGm, 16);

        // AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(uGm);
        // Copy tmpGm to vtGm
        printf("before copyMatrix\n");
        singleDumpTensor(tmpGm, 16);
        CopyMatrix(tmpGm, vtGm, LDN, LDN, LDN, LDN);
        printf("after updateVt\n");
        singleDumpTensor(vtGm, 16);

        return;
    }
    __aicore__ inline void CopyMatrix(GlobalTensor<float> &src, GlobalTensor<float> &dst, uint16_t src_n, uint16_t dst_n, uint16_t copyM, uint16_t copyN)
    {
        NotParallelQuiter;
        const uint8_t padLen = copyN % BlockFloatCnt == 0 ? 0 : BlockFloatCnt - copyN % BlockFloatCnt;
        const uint32_t ttl = copyN + padLen;
        DataCopyExtParams copyInParams = {1, copyN * sizeOfFloat, 0, 0, 0};
        DataCopyExtParams copyOutParams = {1, copyN * sizeOfFloat, 0, 0, 0};
        DataCopyPadExtParams<float> copyInPadParams = {true, 0, padLen, 0.0f};

        for (int i = 0; i < copyM; i++)
        {
            auto bindLocal = copyBind.AllocTensor<float>();
            DataCopyPad(bindLocal, src[i * src_n], copyInParams, copyInPadParams);
            copyBind.EnQue(bindLocal);

            bindLocal = copyBind.DeQue<float>();
            DataCopyPad(dst[i * dst_n], bindLocal, copyOutParams);
            copyBind.FreeTensor(bindLocal);
        }
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

private:
    uint16_t LDM, LDN;
    // in the beginning u and vt are orthogonal matrix, generated by bidiagonalization
    // qt and wt are orthogonal matrix, generated by BDC
    // altogether,the singular matrix are uq,wtvt which
    // would in the end be stored in uGm and vtGm
    // d would be used to store singular values,and e would store the intermediate z;
    GlobalTensor<float> tmpGm, uGm, vtGm, dGm, eGm, qtGm, wtGm;
    GlobalTensor<uint32_t> idxqGm;
    GlobalTensor<uint16_t> svdStackGm;
    BDCMatmulType *mm;
    const uint8_t blockIdx, blockNum;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;
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
