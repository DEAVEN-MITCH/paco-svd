#include "kernel_operator.h"
#include <cmath>
#include "kernel_log.h"
constexpr uint32_t sizeOfFloat = sizeof(float);
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t maxL1FloatSize = 1 << (19 - 2);
constexpr int32_t BlockSize = 32;
constexpr int32_t BlockFloatCnt = BlockSize / sizeOfFloat;
constexpr int32_t SizePerOperation = 256;
constexpr int32_t BlockNumPerOperation = SizePerOperation / BlockSize;
// constexpr uint64_t MASK_PATTERN = 0x0101010101010101ULL;
const AscendC::DataCopyPadExtParams<float> colPadParams = {true, 0, BlockFloatCnt - 1, 0.0f};
// #define _____PIPE_INSIDECLASS
template <class T>
__aicore__ inline constexpr T RoundUpDiv(T x, T div)
{
    return (x + div - 1) / div;
}
__aicore__ inline constexpr int32_t notTilingKGKBSize(int32_t M, int32_t N)
{
    return (1 + 2 * BUFFER_NUM) * M * BlockSize + BlockSize + RoundUpDiv<int32_t>(RoundUpDiv<int32_t>(M, BlockNumPerOperation), BlockFloatCnt) * BlockSize + 40 * 32;
}
__aicore__ inline constexpr int32_t getQueueM()
{
    // return 1024;
    return static_cast<int32_t>(((192 << 10) - 41 * 32) / ((1 + 2 * BUFFER_NUM) * 32 + .5f));
}
constexpr int32_t FixedBufferSize = getQueueM() * BlockSize;
template <bool ifTiling = false>
class Kernel_Golub_Kahan_Bidiagonalization
{
public:
    __aicore__ inline Kernel_Golub_Kahan_Bidiagonalization() : aivIdx(AscendC::GetBlockIdx()), aivNum(AscendC::GetBlockNum()) {}
    // #ifdef _____PIPE_INSIDECLASS
    // __aicore__ inline void Init(const int32_t M, const int32_t N, GM_ADDR a, GM_ADDR u, GM_ADDR vt, GM_ADDR d, GM_ADDR e, GM_ADDR tauq, GM_ADDR taup, GM_ADDR workspace)
    // #else
    __aicore__ inline void Init(const int32_t M, const int32_t N, GM_ADDR a, GM_ADDR u, GM_ADDR vt, GM_ADDR d, GM_ADDR e, GM_ADDR tauq, GM_ADDR taup, GM_ADDR workspace, AscendC::TPipe &pipe)
    // #endif
    {
        // if (AscendC::GetBlockIdx() != 0)
        // return;
        m_ = M;
        n_ = N;
        ASSERT(M >= N && "in Init, we should have M>=N");

        // 设置全局内存缓冲区
        aGm.SetGlobalBuffer((__gm__ float *)a, M * N);
        uGm.SetGlobalBuffer((__gm__ float *)u, M * M);
        vtGm.SetGlobalBuffer((__gm__ float *)vt, N * N);
        dGm.SetGlobalBuffer((__gm__ float *)d, N);
        eGm.SetGlobalBuffer((__gm__ float *)e, N - 1);
        tauqGm.SetGlobalBuffer((__gm__ float *)tauq, N);
        taupGm.SetGlobalBuffer((__gm__ float *)taup, N - 1);

        // 设置同步工作空间
        gmWorkspace.SetGlobalBuffer((__gm__ int32_t *)workspace, aivNum * 32);
        pipe.InitBuffer(ubWorkspaceBuf, aivNum * 32);
        ubWorkspace = ubWorkspaceBuf.Get<int32_t>();

        initUV();

        // 初始化管道缓冲区
        if constexpr (!ifTiling)
        {
            pipe.InitBuffer(inQueue, BUFFER_NUM, M * BlockSize);
            pipe.InitBuffer(outQueue, BUFFER_NUM, M * BlockSize);
            pipe.InitBuffer(houseVecBuf, M * BlockSize);
            pipe.InitBuffer(workLocalBuf, RoundUpDiv<int32_t>(RoundUpDiv<int32_t>(M, BlockNumPerOperation), BlockFloatCnt) * BlockSize);
        }
        else
        {
            pipe.InitBuffer(inQueue, BUFFER_NUM, FixedBufferSize);
            pipe.InitBuffer(outQueue, BUFFER_NUM, FixedBufferSize);
            pipe.InitBuffer(houseVecBuf, FixedBufferSize);
            pipe.InitBuffer(workLocalBuf, RoundUpDiv<int32_t>(RoundUpDiv<int32_t>(getQueueM(), BlockNumPerOperation), BlockFloatCnt) * BlockSize);
        }
        pipe.InitBuffer(scalarBuf, sizeOfFloat);

        houseVec = houseVecBuf.Get<float>();
        scalar = scalarBuf.Get<float>();
        workLocal = workLocalBuf.Get<float>();
    }
    __aicore__ inline void Process()
    {
        // 主循环：对每一列和行进行Householder变换
        for (int32_t i = 0; i < n_; i++)
        {
            // PrintDebugMessage("ith column transform: %d\n", i);
            if (i < n_)
            {
                // 计算列的Householder变换
                ComputeColumnHouseholderV2(i);
                // because we use scalar to modify GM such as dGm, we need to clean the cache of uGm
                AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(uGm);
                AscendC::SyncAll<true>(gmWorkspace, ubWorkspace);
                // PrintDebugMessage("after compute column householder: %d\n", i);
                // 应用列变换到剩余的矩阵
                ApplyColumnTransformV2(i);
                // no Gm modified by scalar, so we don't need to clean the cache of uGm
                AscendC::SyncAll<true>(gmWorkspace, ubWorkspace);
            }
            // PrintDebugMessage("ith column transform: %d\n", i);
            // PrintDebugMessage("ith row transform: %d\n", i);
            if (i < n_ - 1)
            {
                // 计算行的Householder变换
                ComputeRowHouseholderV2(i);
                // because we use scalar to modify GM such as eGm, we need to clean the cache of vtGm
                AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(vtGm);
                AscendC::SyncAll<true>(gmWorkspace, ubWorkspace);
                // 应用行变换到剩余的矩阵
                ApplyRowTransformV2(i);
                // no Gm modified by scalar, so we don't need to clean the cache of vtGm
                AscendC::SyncAll<true>(gmWorkspace, ubWorkspace);
            }
            // PrintDebugMessage("ith row transform: %d\n", i);
        }

        GetUVt();
    }

private:
    __aicore__ inline void ComputeColumnHouseholderV2(int32_t i)
    {
        ASSERT(i < n_ && "in ComputeColumnHouseholder, i should be less than n_");
        if (i % aivNum != aivIdx)
            return;
        const uint16_t len = m_ - i;
        const uint32_t aGmStart = i * n_ + i;
        if (len <= 1)
        {
            // a scalar vector, no need to calc,beta=0;
            tauqGm(i) = 0;
            dGm(i) = aGm(aGmStart);
            return;
        }
        const uint32_t ttl = len * BlockFloatCnt;
        auto tau = tauqGm[i];
        auto deGm = dGm[i];
        if constexpr (ifTiling)
        {
            ComputeHouseholderTiling<true>(ttl, aGmStart, len, colPadParams, tau, deGm);
        }
        else
        {
            const AscendC::DataCopyExtParams copyInExtParams = {len, sizeOfFloat, (n_ - 1) * sizeOfFloat, 0, 0};
            const AscendC::DataCopyExtParams copyOutExtParams = {len, sizeOfFloat, 0, (n_ - 1) * sizeOfFloat, 0};
            ComputeHouseholder(ttl, aGmStart, colPadParams, copyInExtParams, copyOutExtParams, tau, deGm);
        }
    }
    __aicore__ inline void ComputeRowHouseholderV2(int32_t i)
    {
        if (i % aivNum != aivIdx)
            return;
        ASSERT(i < n_ - 1 && "in ComputeRowHouseholder, i should be less than n_ - 1");
        const uint16_t len = n_ - i - 1;
        const uint32_t aGmStart = i * n_ + i + 1;
        if (len <= 1)
        {
            // a scalar vector, no need to calc,beta=0;
            taupGm(i) = 0;
            eGm(i) = aGm(aGmStart);
            return;
        }
        const uint8_t padLen = len % BlockFloatCnt == 0 ? 0 : BlockFloatCnt - len % BlockFloatCnt;
        const uint32_t ttl = len + padLen;
        const AscendC::DataCopyPadExtParams<float> rowPadParams = {true, 0, padLen, 0.0f};
        auto tau = taupGm[i];
        auto deGm = eGm[i];
        if constexpr (ifTiling)
        {
            ComputeHouseholderTiling<false>(ttl, aGmStart, len, rowPadParams, tau, deGm);
        }
        else
        {
            const AscendC::DataCopyExtParams copyInExtParams = {1, len * sizeOfFloat, 0, 0, 0};
            const AscendC::DataCopyExtParams copyOutExtParams = {1, len * sizeOfFloat, 0, 0, 0};
            ComputeHouseholder(ttl, aGmStart, rowPadParams, copyInExtParams, copyOutExtParams, tau, deGm);
        }
    }

    __aicore__ inline void ComputeHouseholder(const uint32_t ttl, const uint32_t aGmStart,
                                              const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
                                              const AscendC::DataCopyExtParams &copyInExtParams,
                                              const AscendC::DataCopyExtParams &copyOutExtParams,
                                              AscendC::GlobalTensor<float> &tau,
                                              AscendC::GlobalTensor<float> &deGm)
    {
        auto x1 = aGm(aGmStart);
        float sigma;
        outputTensor = outQueue.AllocTensor<float>();
        ComputeSigma<true, true, true>(ttl, aGmStart, copyInPadParams, copyInExtParams, sigma);
        if (sigma == 0)
        {
            tau(0) = 0;
            deGm(0) = x1;
            outQueue.FreeTensor(outputTensor);
        }
        else
        {
            auto miu = std::sqrt(x1 * x1 + sigma);
            // PrintDebugMessage("x1:%f,sigma:%f,miu:%f\n", x1, sigma, miu);

            deGm(0) = miu;
            float v1;
            if (x1 <= 0)
            {
                v1 = x1 - miu;
            }
            else
            {
                v1 = -sigma / (x1 + miu);
            }
            float v1sq = v1 * v1, v1_inv = 1.0f / v1;
            tau(0) = 2 * v1sq / (sigma + v1sq);
            ComputeHouseVec<true, true, true>(ttl, aGmStart, copyInPadParams, copyInExtParams, copyOutExtParams, v1_inv, miu);
        }
    }

    template <bool isColumn, bool isFirstLoop, bool isTailLoop = false>
    __aicore__ inline void ComputeSigma(const uint16_t ttl, const uint32_t aGmStart,
                                        const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
                                        const AscendC::DataCopyExtParams &copyInExtParams,
                                        float &sigma)
    {
        LdVecTiling<isColumn, isFirstLoop, isTailLoop>(ttl, aGmStart, aGm, copyInPadParams, copyInExtParams);

        // Compute
        inputTensor = inQueue.DeQue<float>();
        // copy the first inputTensor to outputTensor
        if constexpr (isFirstLoop)
        {
            AscendC::DataCopy(outputTensor, inputTensor, ttl);
            AscendC::Duplicate(inputTensor, 0.0f, 1); // to calculate x[2:len]Tx[2:len],first put the first element to 0.0f
        }
        AscendC::Mul(houseVec, inputTensor, inputTensor, ttl);
        AscendC::ReduceSum(inputTensor, houseVec, inputTensor[BlockFloatCnt], ttl);
        if constexpr (isFirstLoop)
        {
            sigma = inputTensor(0);
        }
        else
        {
            sigma += inputTensor(0);
        }
        // DumpTensor(houseVec, 5, RoundUpDiv(1024, 32) * 32);
        // PrintDebugMessage("ttl:%dsigma:%f\n", ttl, sigma);

        inQueue.FreeTensor(inputTensor);
    }
    template <bool isColumn, bool isFirstLoop, bool isTailLoop = false>
    __aicore__ inline void ComputeHouseVec(const uint16_t ttl,
                                           const uint32_t aGmStart,
                                           const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
                                           const AscendC::DataCopyExtParams &copyInExtParams,
                                           const AscendC::DataCopyExtParams &copyOutExtParams,
                                           const float v1_inv,
                                           const float miu)
    {
        if constexpr (!isFirstLoop)
        {
            LdVecTiling<isColumn, isFirstLoop, isTailLoop>(ttl, aGmStart, aGm, copyInPadParams, copyInExtParams);
            // 计算
            inputTensor = inQueue.DeQue<float>();
            outputTensor = outQueue.AllocTensor<float>();
            AscendC::Muls(outputTensor, inputTensor, v1_inv, ttl);
            inQueue.FreeTensor(inputTensor);
        }
        else
        {
            AscendC::Muls(outputTensor, outputTensor, v1_inv, ttl);
            AscendC::Duplicate(outputTensor, miu, 1);
        }

        outQueue.EnQue(outputTensor);

        // 拷贝输出
        outputTensor = outQueue.DeQue<float>();

        AscendC::DataCopyPad(aGm[aGmStart], outputTensor, copyOutExtParams);

        outQueue.FreeTensor(outputTensor);
    }

    template <bool isColumn>
    __aicore__ inline void ComputeHouseholderTiling(const uint32_t ttl, const uint32_t aGmStart,
                                                    const uint16_t effectiveLen,
                                                    const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
                                                    AscendC::GlobalTensor<float> &tau,
                                                    AscendC::GlobalTensor<float> &deGm)
    {

        const int tailBytes = ttl * sizeOfFloat % FixedBufferSize; // 32 B aligned
        const int formerLoopNum = ttl * sizeOfFloat / FixedBufferSize;
        const bool onlyOneLoop = formerLoopNum == 0;
        if (onlyOneLoop)
        {

            // construct the copyInExtParams and copyOutExtParams,then forward to ComputeHouseholder
            if constexpr (isColumn)
            {
                const AscendC::DataCopyExtParams copyInExtParams = {effectiveLen, sizeOfFloat, (n_ - 1) * sizeOfFloat, 0, 0};
                const AscendC::DataCopyExtParams copyOutExtParams = {effectiveLen, sizeOfFloat, 0, (n_ - 1) * sizeOfFloat, 0};
                ComputeHouseholder(ttl, aGmStart, copyInPadParams, copyInExtParams, copyOutExtParams, tau, deGm);
            }
            else
            {
                const AscendC::DataCopyExtParams copyInExtParams = {1, effectiveLen * sizeOfFloat, 0, 0, 0};
                const AscendC::DataCopyExtParams copyOutExtParams = {1, effectiveLen * sizeOfFloat, 0, 0, 0};
                ComputeHouseholder(ttl, aGmStart, copyInPadParams, copyInExtParams, copyOutExtParams, tau, deGm);
            }
            return;
        }
        // more than one loop
        AscendC::DataCopyExtParams copyInExtParams, copyOutExtParams;
        float x1 = aGm(aGmStart), sigma;
        // Compute sigma first
        outputTensor = outQueue.AllocTensor<float>();
        // the frist loop
        //   Copy in
        uint32_t subTtl, aGmStartStridePerFormerLoop, tailTtl;
        subTtl = FixedBufferSize / sizeOfFloat;
        tailTtl = tailBytes / sizeOfFloat;
        uint32_t tailParam;
        if constexpr (isColumn)
        {
            copyInExtParams.blockCount = getQueueM();
            copyInExtParams.blockLen = sizeOfFloat;
            copyInExtParams.srcStride = (n_ - 1) * sizeOfFloat;
            copyInExtParams.dstStride = 0;
            copyOutExtParams.blockCount = getQueueM();
            copyOutExtParams.blockLen = sizeOfFloat;
            copyOutExtParams.srcStride = 0;
            copyOutExtParams.dstStride = (n_ - 1) * sizeOfFloat;
            aGmStartStridePerFormerLoop = getQueueM() * n_;
            tailParam = tailTtl / BlockFloatCnt;
        }
        else
        {
            copyInExtParams.blockCount = 1;
            copyInExtParams.blockLen = FixedBufferSize;
            copyInExtParams.srcStride = 0;
            copyInExtParams.dstStride = 0;
            copyOutExtParams.blockCount = 1;
            copyOutExtParams.blockLen = FixedBufferSize;
            copyOutExtParams.srcStride = 0;
            copyOutExtParams.dstStride = 0;
            aGmStartStridePerFormerLoop = subTtl;
            tailParam = tailTtl + effectiveLen - ttl;
        }
        ComputeSigma<isColumn, true>(subTtl, aGmStart, copyInPadParams, copyInExtParams, sigma);
        // formerLoopNum -1 loop
        for (int32_t i = 1; i < formerLoopNum; i++)
        {
            ComputeSigma<isColumn, false>(subTtl, aGmStart + aGmStartStridePerFormerLoop * i, copyInPadParams, copyInExtParams, sigma);
        }
        // tailLoop
        if (tailBytes > 0)
        {
            if constexpr (isColumn)
            {
                copyInExtParams.blockCount = tailParam;
            }
            else
            {
                copyInExtParams.blockLen = tailParam;
            }

            ComputeSigma<isColumn, false, true>(tailTtl, aGmStart + aGmStartStridePerFormerLoop * formerLoopNum, copyInPadParams, copyInExtParams, sigma);
        }

        // Compute tau and HouseVec
        if (sigma == 0)
        {
            tau(0) = 0;
            deGm(0) = x1;
            outQueue.FreeTensor(outputTensor);
        }
        else
        {
            auto miu = std::sqrt(x1 * x1 + sigma);
            // PrintDebugMessage("x1:%f,sigma:%f,miu:%f\n", x1, sigma, miu);
            deGm(0) = miu;
            float v1;
            if (x1 <= 0)
            {
                v1 = x1 - miu;
            }
            else
            {
                v1 = -sigma / (x1 + miu);
            }
            float v1sq = v1 * v1, v1_inv = 1.0f / v1;
            tau(0) = 2 * v1sq / (sigma + v1sq);
            if constexpr (isColumn)
            {

                copyInExtParams.blockCount = getQueueM();
            }
            else
            {

                copyInExtParams.blockLen = FixedBufferSize;
            }
            ComputeHouseVec<isColumn, true>(subTtl, aGmStart, copyInPadParams, copyInExtParams, copyOutExtParams, v1_inv, miu);
            for (int32_t i = 1; i < formerLoopNum; i++)
            {
                ComputeHouseVec<isColumn, false>(subTtl, aGmStart + aGmStartStridePerFormerLoop * i, copyInPadParams, copyInExtParams, copyOutExtParams, v1_inv, miu);
            }
            if (tailBytes > 0)
            {
                if constexpr (isColumn)
                {
                    copyOutExtParams.blockCount = tailParam;
                    copyInExtParams.blockCount = tailParam;
                }
                else
                {
                    copyOutExtParams.blockLen = tailParam;
                    copyInExtParams.blockLen = tailParam;
                }
                ComputeHouseVec<isColumn, false, true>(tailTtl, aGmStart + aGmStartStridePerFormerLoop * formerLoopNum, copyInPadParams, copyInExtParams, copyOutExtParams, v1_inv, miu);
            }
        }
    }
    __aicore__ inline void ApplyColumnTransformV2(int32_t i)
    {
        // 应用列变换到右侧子矩阵
        const uint16_t len = m_ - i, allJStart = i + 1;
        const auto jFirst = GetJFirst(allJStart);
        if (jFirst >= n_)
            return;
        float beta = tauqGm(i);
        if (beta == 0)
        {
            return;
        }
        const uint32_t ttl = len * BlockFloatCnt;

        if constexpr (!ifTiling)
        {
            // 加载当前houseVec
            const AscendC::DataCopyExtParams copyInExtParams = {len, sizeOfFloat, (n_ - 1) * sizeOfFloat, 0, 0};
            const AscendC::DataCopyExtParams copyOutExtParams = {len, sizeOfFloat, 0, (n_ - 1) * sizeOfFloat, 0};
            LoadHouseVec(ttl, i * n_ + i, colPadParams, copyInExtParams);
            for (int32_t j = jFirst; j < n_; j += aivNum)
            {
                ApplyTransformCore(ttl, i * n_ + j, colPadParams, copyInExtParams, copyOutExtParams, beta, aGm);
            }
        }
        else
        {
            const int tailBytes = ttl * sizeOfFloat % FixedBufferSize;
            const int formerLoopNum = ttl * sizeOfFloat / FixedBufferSize;
            const bool onlyOneLoop = formerLoopNum == 0;
            if (onlyOneLoop)
            {
                // 加载当前houseVec
                const AscendC::DataCopyExtParams copyInExtParams = {len, sizeOfFloat, (n_ - 1) * sizeOfFloat, 0, 0};
                const AscendC::DataCopyExtParams copyOutExtParams = {len, sizeOfFloat, 0, (n_ - 1) * sizeOfFloat, 0};
                LoadHouseVec(ttl, i * n_ + i, colPadParams, copyInExtParams);
                for (int32_t j = jFirst; j < n_; j += aivNum)
                {
                    ApplyTransformCore(ttl, i * n_ + j, colPadParams, copyInExtParams, copyOutExtParams, beta, aGm);
                }
            }
            else
            {
                for (int32_t j = jFirst; j < n_; j += aivNum)
                {
                    LoadHouseVecAndApplyTransformCoreTiling<true>(ttl, len, i * n_ + j, i * n_ + i, colPadParams, beta, aGm, tailBytes, formerLoopNum, n_);
                }
            }
        }
    }
    __aicore__ inline void ApplyRowTransformV2(int32_t i)
    {
        // 应用行变换到下方子矩阵
        const uint16_t len = n_ - i - 1;
        const uint16_t allJStart = i + 1;
        const auto jFirst = GetJFirst(allJStart);
        if (jFirst >= m_)
            return;
        const uint8_t padLen = len % BlockFloatCnt == 0 ? 0 : BlockFloatCnt - len % BlockFloatCnt;
        const uint32_t ttl = len + padLen;
        float beta = taupGm(i);
        if (beta == 0)
        {
            return;
        }
        AscendC::DataCopyPadExtParams<float> rowPadParams = {true, 0, padLen, 0.0f};
        if constexpr (!ifTiling)
        {
            // 加载当前houseVec
            const AscendC::DataCopyExtParams copyInExtParams = {1, len * sizeOfFloat, 0, 0, 0};
            const AscendC::DataCopyExtParams copyOutExtParams = {1, len * sizeOfFloat, 0, 0, 0};
            LoadHouseVec(ttl, i * n_ + i + 1, rowPadParams, copyInExtParams);
            for (int32_t j = jFirst; j < m_; j += aivNum)
            {
                ApplyTransformCore(ttl, j * n_ + i + 1, rowPadParams, copyInExtParams, copyOutExtParams, beta, aGm);
            }
        }
        else
        {
            const int tailBytes = ttl * sizeOfFloat % FixedBufferSize;
            const int formerLoopNum = ttl * sizeOfFloat / FixedBufferSize;
            const bool onlyOneLoop = formerLoopNum == 0;
            if (onlyOneLoop)
            {
                // 加载当前houseVec
                const AscendC::DataCopyExtParams copyInExtParams = {1, len * sizeOfFloat, 0, 0, 0};
                const AscendC::DataCopyExtParams copyOutExtParams = {1, len * sizeOfFloat, 0, 0, 0};
                LoadHouseVec(ttl, i * n_ + i + 1, rowPadParams, copyInExtParams);
                for (int32_t j = jFirst; j < m_; j += aivNum)
                {
                    ApplyTransformCore(ttl, j * n_ + i + 1, rowPadParams, copyInExtParams, copyOutExtParams, beta, aGm);
                }
            }
            else
            {
                for (int32_t j = jFirst; j < m_; j += aivNum)
                {
                    LoadHouseVecAndApplyTransformCoreTiling<false>(ttl, len, j * n_ + i + 1, i * n_ + i + 1, rowPadParams, beta, aGm, tailBytes, formerLoopNum, n_);
                }
            }
        }
    }

    __aicore__ inline void ApplyTransformCore(
        const uint32_t ttl,
        const uint32_t targetGmStart,
        const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
        const AscendC::DataCopyExtParams &copyInExtParams,
        const AscendC::DataCopyExtParams &copyOutExtParams,
        const float beta,
        AscendC::GlobalTensor<float> &targetGm)
    {
        // copy in
        inputTensor = inQueue.AllocTensor<float>();
        AscendC::DataCopyPad(inputTensor, targetGm[targetGmStart], copyInExtParams, copyInPadParams);
        inQueue.EnQue(inputTensor);

        // compute
        inputTensor = inQueue.DeQue<float>();
        outputTensor = outQueue.AllocTensor<float>();
        // row*v,use row as demonstrative example
        AscendC::Mul(outputTensor, inputTensor, houseVec, ttl);
        AscendC::ReduceSum(scalar, outputTensor, workLocal, ttl);
        // beta*row*v
        AscendC::Muls(scalar, scalar, beta, 1);
        float coeff = scalar(0);
        AscendC::Muls(outputTensor, houseVec, coeff, ttl);
        // row - beta*row*v*vT
        AscendC::Sub(outputTensor, inputTensor, outputTensor, ttl);

        inQueue.FreeTensor(inputTensor);
        outQueue.EnQue(outputTensor);

        // copy out
        outputTensor = outQueue.DeQue<float>();
        AscendC::DataCopyPad(targetGm[targetGmStart], outputTensor, copyOutExtParams);
        outQueue.FreeTensor(outputTensor);
    }
    template <bool isColumn, bool isFirstLoop, bool isTailLoop = false>
    __aicore__ inline void LdVecTiling(
        const uint32_t ttl,
        const uint32_t targetGmStart,
        AscendC::GlobalTensor<float> &targetGm,
        const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
        const AscendC::DataCopyExtParams &copyInExtParams)
    {
        inputTensor = inQueue.AllocTensor<float>();
        if constexpr (isColumn || isTailLoop)
        {
            AscendC::DataCopyPad(inputTensor, targetGm[targetGmStart], copyInExtParams, copyInPadParams);
        }
        else
        {
            AscendC::DataCopyPad(inputTensor, targetGm[targetGmStart], copyInExtParams, {});
        }

        inQueue.EnQue(inputTensor);
    }
    template <bool isColumn, bool isFirstLoop, bool isTailLoop = false>
    __aicore__ inline void LdHouseVecTiling(
        const uint32_t ttl,
        const uint32_t aGmStart,
        const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
        const AscendC::DataCopyExtParams &copyInExtParams)
    {
        // copy in
        LdVecTiling<isColumn, isFirstLoop, isTailLoop>(ttl, aGmStart, aGm, copyInPadParams, copyInExtParams);

        // compute
        inputTensor = inQueue.DeQue<float>();
        AscendC::DataCopy(houseVec, inputTensor, ttl);
        if constexpr (isFirstLoop)
        {
            AscendC::Duplicate(houseVec, 1.0f, 1);
        }
        inQueue.FreeTensor(inputTensor);
    }

    template <bool isColumn, bool isFirstLoop, bool isTailLoop = false>
    __aicore__ inline void CalculateCoeffTiling(
        const uint32_t ttl,
        const uint32_t targetGmStart,
        const uint32_t aGmStart,
        const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
        const AscendC::DataCopyExtParams &copyInParams,
        const AscendC::DataCopyExtParams &copyInHouseVecParams,
        AscendC::GlobalTensor<float> &targetGm,
        float &coeff)
    {
        LdHouseVecTiling<isColumn, isFirstLoop, isTailLoop>(ttl, aGmStart, copyInPadParams, copyInHouseVecParams);
        LdVecTiling<isColumn, isFirstLoop, isTailLoop>(ttl, targetGmStart, targetGm, copyInPadParams, copyInParams);
        inputTensor = inQueue.DeQue<float>();
        AscendC::Mul(outputTensor, inputTensor, houseVec, ttl);
        AscendC::ReduceSum(scalar, outputTensor, workLocal, ttl);
        coeff += scalar(0);
        inQueue.FreeTensor(inputTensor);
    }
    template <bool isColumn, bool isFirstLoop, bool isTailLoop = false>
    __aicore__ inline void ApplyTransformStage2(
        const uint32_t ttl,
        const uint32_t targetGmStart,
        const uint32_t aGmStart,
        const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
        const AscendC::DataCopyExtParams &copyInParams,
        const AscendC::DataCopyExtParams &copyInHouseVecParams,
        const AscendC::DataCopyExtParams &copyOutExtParams,
        AscendC::GlobalTensor<float> &targetGm,
        const float &coeff)
    {
        LdHouseVecTiling<isColumn, isFirstLoop, isTailLoop>(ttl, aGmStart, copyInPadParams, copyInHouseVecParams);
        LdVecTiling<isColumn, isFirstLoop, isTailLoop>(ttl, targetGmStart, targetGm, copyInPadParams, copyInParams);
        inputTensor = inQueue.DeQue<float>();
        if constexpr (!isFirstLoop)
        {
            outputTensor = outQueue.AllocTensor<float>();
        }
        AscendC::Muls(outputTensor, houseVec, coeff, ttl);
        AscendC::Sub(outputTensor, inputTensor, outputTensor, ttl);
        inQueue.FreeTensor(inputTensor);
        outQueue.EnQue(outputTensor);

        outputTensor = outQueue.DeQue<float>();
        AscendC::DataCopyPad(targetGm[targetGmStart], outputTensor, copyOutExtParams);
        outQueue.FreeTensor(outputTensor);
    }
    template <bool isColumn>
    __aicore__ inline void LoadHouseVecAndApplyTransformCoreTiling(
        const uint16_t ttl,
        const uint32_t effectiveLen,
        const uint32_t targetGmStart,
        const uint32_t aGmStart,
        const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
        const float beta,
        AscendC::GlobalTensor<float> &targetGm,
        const uint32_t tailBytes,
        const uint32_t formerLoopNum,
        const uint16_t targetLDN)
    {
        AscendC::DataCopyExtParams copyInHouseVecExtParams;
        AscendC::DataCopyExtParams copyInExtParams;
        AscendC::DataCopyExtParams copyOutExtParams;

        // 初始化参数
        uint16_t subTtl;
        uint32_t aGmStartStridePerFormerLoop, tailTtl, targetGmStartStridePerFormerLoop;
        subTtl = FixedBufferSize / sizeOfFloat;
        tailTtl = tailBytes / sizeOfFloat;
        uint32_t tailParam;
        if constexpr (isColumn)
        {
            aGmStartStridePerFormerLoop = getQueueM() * n_;
            targetGmStartStridePerFormerLoop = getQueueM() * targetLDN;
            copyInHouseVecExtParams = {getQueueM(), sizeOfFloat, (n_ - 1) * sizeOfFloat, 0, 0};
            copyInExtParams = {getQueueM(), sizeOfFloat, (targetLDN - 1) * sizeOfFloat, 0, 0};
            copyOutExtParams = {getQueueM(), sizeOfFloat, 0, (targetLDN - 1) * sizeOfFloat, 0};
            tailParam = tailTtl / BlockFloatCnt;
        }
        else
        {
            aGmStartStridePerFormerLoop = subTtl;
            targetGmStartStridePerFormerLoop = subTtl;
            copyInHouseVecExtParams = {1, FixedBufferSize, 0, 0, 0};
            copyInExtParams = {1, FixedBufferSize, 0, 0, 0};
            copyOutExtParams = {1, FixedBufferSize, 0, 0, 0};
            tailParam = tailTtl + effectiveLen - ttl;
        }
        outputTensor = outQueue.AllocTensor<float>();

        float coeff = 0.0f;
        // first calculate beta*row*v
        // first loop
        CalculateCoeffTiling<isColumn, true>(subTtl, targetGmStart, aGmStart, copyInPadParams, copyInExtParams, copyInHouseVecExtParams, targetGm, coeff);
        // formerLoopNum - 1
        for (int32_t i = 1; i < formerLoopNum; i++)
        {
            CalculateCoeffTiling<isColumn, false>(subTtl, targetGmStart + i * targetGmStartStridePerFormerLoop, aGmStart + i * aGmStartStridePerFormerLoop, copyInPadParams, copyInExtParams, copyInHouseVecExtParams, targetGm, coeff);
        }
        if (tailBytes > 0)
        {
            if constexpr (isColumn)
            {
                copyInExtParams.blockCount = tailParam;
                copyInHouseVecExtParams.blockCount = tailParam;
            }
            else
            {
                copyInExtParams.blockLen = tailParam;
                copyInHouseVecExtParams.blockLen = tailParam;
            }
            CalculateCoeffTiling<isColumn, false, true>(tailTtl, targetGmStart + formerLoopNum * targetGmStartStridePerFormerLoop, aGmStart + formerLoopNum * aGmStartStridePerFormerLoop, copyInPadParams, copyInExtParams, copyInHouseVecExtParams, targetGm, coeff);
        }
        coeff *= beta;
        // print coeff
        // PrintDebugMessage("coeff:%f\n", coeff);
        // then calculate row - beta*row*v*vT
        if constexpr (isColumn)
        {
            copyInExtParams.blockCount = getQueueM();
            copyInHouseVecExtParams.blockCount = getQueueM();
        }
        else
        {
            copyInExtParams.blockLen = FixedBufferSize;
            copyInHouseVecExtParams.blockLen = FixedBufferSize;
        }
        // first loop
        ApplyTransformStage2<isColumn, true>(subTtl, targetGmStart, aGmStart, copyInPadParams, copyInExtParams, copyInHouseVecExtParams, copyOutExtParams, targetGm, coeff);
        // formerLoopNum-1 loop
        for (int32_t i = 1; i < formerLoopNum; ++i)
        {
            ApplyTransformStage2<isColumn, false>(subTtl, targetGmStart + i * targetGmStartStridePerFormerLoop, aGmStart + i * aGmStartStridePerFormerLoop, copyInPadParams, copyInExtParams, copyInHouseVecExtParams, copyOutExtParams, targetGm, coeff);
        }
        if (tailBytes > 0)
        {
            if constexpr (isColumn)
            {
                copyInExtParams.blockCount = tailParam;
                copyInHouseVecExtParams.blockCount = tailParam;
                copyOutExtParams.blockCount = tailParam;
            }
            else
            {
                copyInExtParams.blockLen = tailParam;
                copyInHouseVecExtParams.blockLen = tailParam;
                copyOutExtParams.blockLen = tailParam;
            }
            ApplyTransformStage2<isColumn, false, true>(tailTtl, targetGmStart + formerLoopNum * targetGmStartStridePerFormerLoop, aGmStart + formerLoopNum * aGmStartStridePerFormerLoop, copyInPadParams, copyInExtParams, copyInHouseVecExtParams, copyOutExtParams, targetGm, coeff);
        }
    }

    __aicore__ inline void LoadHouseVec(
        const uint32_t ttl,
        const uint32_t aGmStart,
        const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
        const AscendC::DataCopyExtParams &copyInExtParams)
    {
        // copy in
        inputTensor = inQueue.AllocTensor<float>();
        AscendC::DataCopyPad(inputTensor, aGm[aGmStart], copyInExtParams, copyInPadParams);
        inQueue.EnQue(inputTensor);

        // compute
        inputTensor = inQueue.DeQue<float>();
        // AscendC::Adds(houseVec, inputTensor, .0f, ttl);
        AscendC::DataCopy(houseVec, inputTensor, ttl);
        AscendC::Duplicate(houseVec, 1.0f, 1);
        inQueue.FreeTensor(inputTensor);
    }
    __aicore__ inline void GetUVt()
    {
        // 刷新Cache，保证uGm、vGm与Cache的一致性,由于之前Process已经刷新过，无需再刷新
        // AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(uGm);
        // get U
        for (int32_t i = n_ - 1; i >= 0; i--)
        {
            // PrintDebugMessage("before ith column transform of U: %d\n", i);
            // the i-th householder vector updates m_-i columns of U
            const uint16_t len = m_ - i;
            auto beta = tauqGm(i);
            if (beta == 0)
            {
                continue;
            }
            const auto jFirst = GetJFirst(i);
            if (jFirst < m_)
            {
                const uint32_t ttl = len * BlockFloatCnt;
                if constexpr (!ifTiling)
                {
                    // 加载当前houseVec
                    const AscendC::DataCopyExtParams copyInExtParams = {len, sizeOfFloat, (n_ - 1) * sizeOfFloat, 0, 0};
                    LoadHouseVec(ttl, i * n_ + i, colPadParams, copyInExtParams);

                    for (int32_t j = jFirst; j < m_; j += aivNum)
                    {
                        const AscendC::DataCopyExtParams copyInExtParams = {len, sizeOfFloat, (m_ - 1) * sizeOfFloat, 0, 0};
                        const AscendC::DataCopyExtParams copyOutExtParams = {len, sizeOfFloat, 0, (m_ - 1) * sizeOfFloat, 0};
                        ApplyTransformCore(ttl, i * m_ + j, colPadParams, copyInExtParams, copyOutExtParams, beta, uGm);
                    }
                }
                else
                {
                    const int tailBytes = ttl * sizeOfFloat % FixedBufferSize;
                    const int formerLoopNum = ttl * sizeOfFloat / FixedBufferSize;
                    if (formerLoopNum == 0)
                    {
                        // 加载当前houseVec
                        const AscendC::DataCopyExtParams copyInExtParams = {len, sizeOfFloat, (n_ - 1) * sizeOfFloat, 0, 0};
                        LoadHouseVec(ttl, i * n_ + i, colPadParams, copyInExtParams);

                        for (int32_t j = jFirst; j < m_; j += aivNum)
                        {
                            const AscendC::DataCopyExtParams copyInExtParams = {len, sizeOfFloat, (m_ - 1) * sizeOfFloat, 0, 0};
                            const AscendC::DataCopyExtParams copyOutExtParams = {len, sizeOfFloat, 0, (m_ - 1) * sizeOfFloat, 0};
                            ApplyTransformCore(ttl, i * m_ + j, colPadParams, copyInExtParams, copyOutExtParams, beta, uGm);
                        }
                    }
                    else
                    {
                        for (int32_t j = jFirst; j < m_; j += aivNum)
                        {
                            LoadHouseVecAndApplyTransformCoreTiling<true>(ttl, len, i * m_ + j, i * n_ + i, colPadParams, beta, uGm, tailBytes, formerLoopNum, m_);
                        }
                    }
                }
            }
            AscendC::SyncAll<true>(gmWorkspace, ubWorkspace);
        }

        // get Vt
        for (int32_t i = n_ - 2; i >= 0; i--)
        {
            // PrintDebugMessage("before ith row transform of Vt: %d\n", i);
            //  the i-th householder vector updates n_-i rows,n-i-1 columns of Vt
            const uint16_t len = n_ - i - 1;
            const uint8_t padLen = len % BlockFloatCnt == 0 ? 0 : BlockFloatCnt - len % BlockFloatCnt;
            const uint32_t ttl = len + padLen;
            auto beta = taupGm(i);
            if (beta == 0)
            {
                continue;
            }
            const auto jFirst = GetJFirst(i);
            if (jFirst < n_)
            {
                AscendC::DataCopyPadExtParams<float> rowPadParams = {true, 0, padLen, 0.0f};
                if constexpr (!ifTiling)
                {
                    // 加载当前houseVec
                    const AscendC::DataCopyExtParams copyInExtParams = {1, len * sizeOfFloat, 0, 0, 0};
                    LoadHouseVec(ttl, i * n_ + i + 1, rowPadParams, copyInExtParams);

                    for (int32_t j = jFirst; j < n_; j += aivNum)
                    {
                        const AscendC::DataCopyExtParams copyInExtParams = {1, len * sizeOfFloat, 0, 0, 0};
                        const AscendC::DataCopyExtParams copyOutExtParams = {1, len * sizeOfFloat, 0, 0, 0};
                        ApplyTransformCore(ttl, j * n_ + i + 1, rowPadParams, copyInExtParams, copyOutExtParams, beta, vtGm);
                    }
                }
                else
                {
                    const int tailBytes = ttl * sizeOfFloat % FixedBufferSize;
                    const int formerLoopNum = ttl * sizeOfFloat / FixedBufferSize;
                    if (formerLoopNum == 0)
                    {
                        // 加载当前houseVec
                        const AscendC::DataCopyExtParams copyInExtParams = {1, len * sizeOfFloat, 0, 0, 0};
                        LoadHouseVec(ttl, i * n_ + i + 1, rowPadParams, copyInExtParams);

                        for (int32_t j = jFirst; j < n_; j += aivNum)
                        {
                            const AscendC::DataCopyExtParams copyInExtParams = {1, len * sizeOfFloat, 0, 0, 0};
                            const AscendC::DataCopyExtParams copyOutExtParams = {1, len * sizeOfFloat, 0, 0, 0};
                            ApplyTransformCore(ttl, j * n_ + i + 1, rowPadParams, copyInExtParams, copyOutExtParams, beta, vtGm);
                        }
                    }
                    else
                    {
                        for (int32_t j = jFirst; j < n_; j += aivNum)
                        {
                            LoadHouseVecAndApplyTransformCoreTiling<false>(ttl, len, j * n_ + i + 1, i * n_ + i + 1, rowPadParams, beta, vtGm, tailBytes, formerLoopNum, n_);
                        }
                    }
                }
            }
            AscendC::SyncAll<true>(gmWorkspace, ubWorkspace);
        }
    }
    __aicore__ inline void initUV()
    {
        // cache 问题，AIV处理前需刷新Cache
        for (int32_t i = aivIdx; i < m_; i += aivNum)
        {
            uGm(i * m_ + i) = 1.0f;
        }
        for (int32_t i = aivIdx; i < n_; i += aivNum)
        {
            vtGm(i * n_ + i) = 1.0f;
        }
    }

private:
    __aicore__ inline uint16_t GetJFirst(const uint16_t allJStart)
    {
        const auto j_floor = allJStart / aivNum * aivNum;
        const auto jFirst = j_floor + aivIdx < allJStart ? j_floor + aivIdx + aivNum : j_floor + aivIdx;
        return jFirst;
    }

    template <typename... Args>
    __aicore__ inline void PrintDebugMessage(Args... args)
    {
        if (aivIdx == 0)
        {
            AscendC::printf(args...);
        }
    }

private:
    uint16_t m_, n_;
    const uint8_t aivNum, aivIdx;
#ifdef _____PIPE_INSIDECLASS
    AscendC::TPipe pipe;
#endif
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> houseVecBuf, scalarBuf, workLocalBuf, ubWorkspaceBuf;
    AscendC::GlobalTensor<float> aGm, uGm, vtGm, dGm, eGm, tauqGm, taupGm;
    AscendC::GlobalTensor<int32_t> gmWorkspace;
    AscendC::LocalTensor<float> houseVec, scalar, workLocal, inputTensor, outputTensor;
    AscendC::LocalTensor<int32_t> ubWorkspace;
};

extern "C" __global__ __aicore__ void upper_bidiagonalization(int M, int N, GM_ADDR a, GM_ADDR u, GM_ADDR vt, GM_ADDR d, GM_ADDR e, GM_ADDR tauq, GM_ADDR taup, GM_ADDR workspace)
{
// #ifndef _____PIPE_INSIDECLASS
#ifdef __DAV_C220_VEC__

    AscendC::TPipe pipe;
    if (auto UBsizeRequired = notTilingKGKBSize(M, N); UBsizeRequired < (192 << 10))
    {
        // if (AscendC::GetBlockIdx() == 0)
        // {
        // AscendC::printf("the UBsizeRequired is %d\n", UBsizeRequired);
        // }
        Kernel_Golub_Kahan_Bidiagonalization<false> kernel;
        kernel.Init(M, N, a, u, vt, d, e, tauq, taup, workspace, pipe);
        // #else
        // Kernel_Golub_Kahan_Bidiagonalization kernel;
        // kernel.Init(M, N, a, u, vt, d, e, tauq, taup, workspace);
        // #endif
        kernel.Process();
    }
    else
    {
        // #ifndef _____PIPE_INSIDECLASS
        Kernel_Golub_Kahan_Bidiagonalization<true> kernel;
        kernel.Init(M, N, a, u, vt, d, e, tauq, taup, workspace, pipe);
        // #else
        // Kernel_Golub_Kahan_Bidiagonalization<true> kernel;
        // kernel.Init(M, N, a, u, vt, d, e, tauq, taup, workspace);
        // #endif
        kernel.Process();
    }
#endif
}
