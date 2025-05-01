#include "kernel_operator.h"
#include <lib/matmul_intf.h>

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
constexpr float EPS = 1e-6;
using namespace AscendC;
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
        // TCubeTiling tiling;
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
}
template <bool ifVecTiling = false, bool ifParallel = false>
class BDC
{
public:
    __aicore__ inline BDC() : blockIdx(AscendC::GetBlockIdx()), blockNum(AscendC::GetBlockNum()) {}
    __aicore__ inline void init(int M, int N, GM_ADDR a, GM_ADDR u, GM_ADDR vt, GM_ADDR d, GM_ADDR e, GM_ADDR q, GM_ADDR wt, GM_ADDR idx, GM_ADDR svdStack, GM_ADDR workspace, SVDTiling *tiling, TPipe &pipe)
    {
        if constexpr (!ifParallel)
        {
            if (blockIdx != 0)
            {
                return;
            }
        }
        ASSERT(M >= N && "M must be greater than or equal to N");
        LDM = M;
        LDN = N;
        svdTiling = tiling;
        svdStackGm.SetGlobalBuffer((__gm__ uint16_t *)svdStack, tiling->stackSize);
        tmpGm.SetGlobalBuffer((__gm__ float *)a, M * N);
        uGm.SetGlobalBuffer((__gm__ float *)u, M * M);
        vtGm.SetGlobalBuffer((__gm__ float *)vt, N * N);
        dGm.SetGlobalBuffer((__gm__ float *)d, N);
        eGm.SetGlobalBuffer((__gm__ float *)e, N - 1);
        qGm.SetGlobalBuffer((__gm__ float *)q, N * N);
        wtGm.SetGlobalBuffer((__gm__ float *)wt, N * N);
        idxqGm.SetGlobalBuffer((__gm__ uint32_t *)idx, N);
        // pipe.InitBuffer(workspaceBuf, blockNum * 32);

        // workspace = workspaceBuf.Get<int32_t>();
    }
    __aicore__ inline void Process()
    {
        if constexpr (!ifParallel)
        {
            if (blockIdx != 0)
            {
                return;
            }
        }
        // reduction requires rotation in the end,may not be worth it
        // simply no reduction ,as sbdsdc does
        initQWt();
        if (LDN == 2)
        {
            if(blockIdx==0)
            {
                compute_2x2_svd(qGm, wtGm, dGm, eGm, idxqGm);
                updateUVt();
            }

            return;
        }
        else if (LDN == 1)
        {
            if(blockIdx==0)
            {
                compute_1x1_svd(qGm, wtGm, dGm, idxqGm);
                updateUVt();
            }

            return;
        }
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
                auto &mergedSubMatrix = getSVDSubmatrixInfo(newStackSize++);
                mergedSubMatrix.start_col = leftSubMatrix.start_col;
                mergedSubMatrix.end_col = rightSubMatrix.end_col;
                MergeSubMatrix(leftSubMatrix, rightSubMatrix);
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
            updateUVt();
        }
    }

private:
    __aicore__ inline void initQWt()
    {
#ifdef __DAV_C220_VEC__
        // use aiv's scalars only
        //  初始化q和wt为单位矩阵
        if constexpr (!ifParallel)
        {
            for (auto i = 0; i < LDN; i++)
            {
                qGm(i * LDN + i) = 1.0f;
                wtGm(i * LDN + i) = 1.0f;
            }
        }
        else
        {
            for (int32_t i = blockIdx; i < LDN; i += blockNum)
            {
                qGm(i * LDN + i) = 1.0f;
                wtGm(i * LDN + i) = 1.0f;
            }
        }
#endif
    }

    __aicore__ inline void MergeSubMatrix(const SVDSubmatrixInfo &leftSubMatrix, const SVDSubmatrixInfo &rightSubMatrix)
    {
        return;
    }

    __aicore__ inline void compute_base_case_svd(const SVDSubmatrixInfo &subMatrix)
    {
        const auto idx_start = subMatrix.start_col;
        const auto colNum = subMatrix.end_col - idx_start + 1;
        const auto rowNum = subMatrix.end_col == LDN ? colNum : colNum - 1;
        GlobalTensor<float> q = qGm[idx_start * LDN + idx_start];
        GlobalTensor<float> wt = wtGm[idx_start * LDN + idx_start];
        GlobalTensor<float> d = dGm[idx_start];
        GlobalTensor<uint32_t> idxq = idxqGm[idx_start];
        // allow e to be valid in case 1x1
        GlobalTensor<float> e = idx_start == LDN - 1 ? dGm[idx_start] : eGm[idx_start];
        if (colNum == 3)
        {
            compute_2x3_svd(q, wt, d, e, idxq);
        }
        else if (colNum == 2 && rowNum == 2)
        {
            compute_2x2_svd(q, wt, d, e, idxq);
        }
        else if (colNum == 2 && rowNum == 1)
        {
            compute_1x2_svd(q, wt, d, e, idxq);
        }
        else if (colNum == 1 && rowNum == 1)
        {
            compute_1x1_svd(q, wt, d, idxq);
        }
    }
    __aicore__ inline void compute_2x3_svd(GlobalTensor<float> &q, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<float> &e, GlobalTensor<uint32_t> &idxq)
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
            q(0) = 0;
            q(1) = 1;
            q(LDN) = 1;
            q(LDN + 1) = 0;
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
            // q is unit
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
            q(0) = a12 / sq;
            q(1) = a22 / sq;
            q(LDN) = a22 / sq;
            q(LDN + 1) = -a12 / sq;
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
            q(0) = u1 / normu;
            q(LDN) = u2 / normu;
            wt(0) = v1 / normv;
            wt(1) = v2 / normv;
            wt(2) = v3 / normv;
            u1 = sigma2 * sigma2 - m22, v1 = a11 * u1, v2 = a12 * (sigma2 * sigma2 - a23 * a23);
            normu = sqrt(u1 * u1 + u2 * u2), normv = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
            q(1) = u1 / normu;
            q(LDN + 1) = u2 / normu;
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
    __aicore__ inline void compute_2x2_svd(GlobalTensor<float> &q, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<float> &e, GlobalTensor<uint32_t> &idxq)
    {
        float a11 = d(0), a12 = e(0), a22 = d(1);

        idxq(0) = 0;
        idxq(1) = 1;
        if (a11 == 0 && a12 == 0 && a22 == 0)
        {
            // rank 0
            d(0) = 0;
            d(1) = 0;
            // q(0) = 1.0f;
            // q(LDN + 1) = 1.0f;
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
            // q(0) = 1.0f;
            // q(LDN + 1) = 1.0f;
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
            q(0) = a12 / sq;
            q(1) = a22 / sq;
            q(LDN) = a22 / sq;
            q(LDN + 1) = -a12 / sq;
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
                q(0) = sign(1.0f, a11);
                q(LDN + 1) = sign(1.0f, a22);
            }
            else
            {
                d(0) = fabs(a22);
                d(1) = fabs(a11);
                q(0) = 0.0f;
                q(1) = sign(1.0f, a11);
                q(LDN) = sign(1.0f, a22);
                q(LDN + 1) = 0.0f;
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
            q(0) = u1 / normu;
            q(LDN) = u2 / normu;
            v1 = sigma2 * sigma2 - a12 * a12 - m22, u1 = sigma2 * sigma2 - m22;
            normv = sqrt(v1 * v1 + v2 * v2), normu = sqrt(u1 * u1 + u2 * u2);
            wt(LDN) = v1 / normv;
            wt(LDN + 1) = v2 / normv;
            q(1) = u1 / normu;
            q(LDN + 1) = u2 / normu;
            return;
        }
    }
    __aicore__ inline void compute_1x2_svd(GlobalTensor<float> &q, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<float> &e, GlobalTensor<uint32_t> &idxq)
    {
        float a11 = d(0), a12 = e(0);
        float sq = sqrt(a11 * a11 + a12 * a12);
        idxq(0) = 0;
        d(0) = sq;
        // q(0) = 1.0f;
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
    __aicore__ inline void compute_1x1_svd(GlobalTensor<float> &q, GlobalTensor<float> &wt, GlobalTensor<float> &d, GlobalTensor<uint32_t> &idxq)
    {
        idxq(0) = 0;
        float a11 = d(0);
        q(0) = sign(1.0f, a11);
        d(0) = fabs(a11);
        // wt(0) = 1.0f;
        // no need to set w to 1.0f because it initializes to unit matrix
    }

    __aicore__ inline void updateUVt()
    {
        // update UVt from Q Wt
        // LDN columns of U  are updated
        // all rows of Vt are updated
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
    // q and wt are orthogonal matrix, generated by BDC
    // altogether,the singular matrix are uq,wtvt which
    // would in the end be stored in uGm and vtGm
    // d would be used to store singular values,and e would store the intermediate z;
    GlobalTensor<float> tmpGm, uGm, vtGm, dGm, eGm, qGm, wtGm;
    GlobalTensor<uint32_t> idxqGm;
    GlobalTensor<uint16_t> svdStackGm;
    const uint8_t blockIdx, blockNum;
    SVDTiling *svdTiling;
};

extern "C" __global__ __aicore__ void svd_DC(int M, int N, GM_ADDR a, GM_ADDR u, GM_ADDR vt, GM_ADDR d, GM_ADDR e, GM_ADDR q, GM_ADDR wt, GM_ADDR idx, GM_ADDR workspace, GM_ADDR tilingGM)
{
#ifdef __DAV_C220_VEC__
    TPipe pipe;
    BDC bdc;
    SVDTiling tiling;
    GM_ADDR svdStack;
    CopyTiling(&tiling, &svdStack, tilingGM);
    bdc.init(M, N, a, u, vt, d, e, q, wt, idx, svdStack, workspace, &tiling, pipe);

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
#endif
}
