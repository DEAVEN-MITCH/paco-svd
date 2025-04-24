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
    return static_cast<int32_t>(((192 << 10) - 41 * 32) / ((1 + 2 * BUFFER_NUM) * 32 + .5));
}
constexpr int32_t FixedBufferSize = getQueueM() * BlockSize;
template <bool ifTiling = false>
class Kernel_Golub_Kahan_Bidiagonalization
{
public:
    __aicore__ inline Kernel_Golub_Kahan_Bidiagonalization() : aivIdx(AscendC::GetBlockIdx()), aivNum(AscendC::GetBlockNum()) {}
#ifdef _____PIPE_INSIDECLASS
    __aicore__ inline void Init(const int32_t M, const int32_t N, GM_ADDR a, GM_ADDR u, GM_ADDR vt, GM_ADDR d, GM_ADDR e, GM_ADDR tauq, GM_ADDR taup, GM_ADDR workspace)
#else
    __aicore__ inline void Init(const int32_t M, const int32_t N, GM_ADDR a, GM_ADDR u, GM_ADDR vt, GM_ADDR d, GM_ADDR e, GM_ADDR tauq, GM_ADDR taup, GM_ADDR workspace, AscendC::TPipe &pipe)
#endif
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
            // AscendC::printf("ith column transform: %d\n", i);
            if (i < n_)
            {
                // 计算列的Householder变换
                ComputeColumnHouseholderV2(i);
                // because we use scalar to modify GM such as dGm, we need to clean the cache of uGm
                AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(uGm);
                AscendC::SyncAll<true>(gmWorkspace, ubWorkspace);
                // 应用列变换到剩余的矩阵
                ApplyColumnTransformV2(i);
                // no Gm modified by scalar, so we don't need to clean the cache of uGm
                AscendC::SyncAll<true>(gmWorkspace, ubWorkspace);
            }

            // AscendC::printf("ith row transform: %d\n", i);
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
            ComputeHouseholderTiling<true>(ttl, aGmStart, colPadParams, copyInExtParams, copyOutExtParams, tau, deGm);
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
            ComputeHouseholderTiling<false>(ttl, aGmStart, rowPadParams, copyInExtParams, copyOutExtParams, tau, deGm);
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
        auto x1 = aGm[aGmStart];

        // Copy in
        inputTensor = inQueue.AllocTensor<float>();
        AscendC::DataCopyPad(inputTensor, aGm[aGmStart], copyInExtParams, copyInPadParams);
        inQueue.EnQue(inputTensor);

        // Compute
        inputTensor = inQueue.DeQue<float>();
        outputTensor = outQueue.AllocTensor<float>();

        // copy inputTensor to outputTensor
        // AscendC::Adds(outputTensor, inputTensor, 0.0f, ttl);
        AscendC::DataCopy(outputTensor, inputTensor, ttl);
        AscendC::Duplicate(inputTensor, 0.0f, 1); // to calculate x[2:len]Tx[2:len],first put the first element to 0.0f
        AscendC::Mul(houseVec, inputTensor, inputTensor, ttl);
        AscendC::ReduceSum(inputTensor, houseVec, inputTensor[BlockFloatCnt], ttl);
        auto sigma = inputTensor(0);
        inQueue.FreeTensor(inputTensor);
        if (sigma == 0)
        {
            tau(0) = 0;
            deGm(0) = x1;
            outQueue.FreeTensor(outputTensor);
        }
        else
        {
            auto miu = std::sqrt(x1 * x1 + sigma);
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
            AscendC::Muls(outputTensor, outputTensor, v1_inv, ttl);
            // AscendC::Adds(houseVec, outputTensor, 0.0f, ttl);
            // AscendC::DataCopy(houseVec, outputTensor, ttl);
            // AscendC::Duplicate(houseVec, 1.0f, 1);
            AscendC::Duplicate(outputTensor, miu, 1);
            outQueue.EnQue(outputTensor);

            // Copy out
            outputTensor = outQueue.DeQue<float>();
            AscendC::DataCopyPad(aGm[aGmStart], outputTensor, copyOutExtParams);
            outQueue.FreeTensor(outputTensor);
        }
    }
    template <bool isColumn>
    __aicore__ inline void ComputeHouseholderTiling(const uint32_t ttl, const uint32_t aGmStart,
                                                    const uint32_t effectiveLen,
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
                const AscendC::DataCopyExtParams copyInExtParams = {len, sizeOfFloat, (n_ - 1) * sizeOfFloat, 0, 0};
                const AscendC::DataCopyExtParams copyOutExtParams = {len, sizeOfFloat, 0, (n_ - 1) * sizeOfFloat, 0};
                ComputeHouseholder(ttl, aGmStart, copyInPadParams, copyInExtParams, copyOutExtParams, tau, deGm);
            }
            else
            {
                const AscendC::DataCopyExtParams copyInExtParams = {1, len * sizeOfFloat, 0, 0, 0};
                const AscendC::DataCopyExtParams copyOutExtParams = {1, len * sizeOfFloat, 0, 0, 0};
                ComputeHouseholder(ttl, aGmStart, copyInPadParams, copyInExtParams, copyOutExtParams, tau, deGm);
            }
            return;
        }
        // more than one loop
        AscendC::DataCopyExtParams copyInExtParams, copyOutExtParams;
        /*  __aicore__ DataCopyExtParams()
    {
        blockCount = DEFAULT_DATA_COPY_NBURST;
        blockLen = 0;
        srcStride = DEFAULT_DATA_COPY_STRIDE;
        dstStride = DEFAULT_DATA_COPY_STRIDE;
        rsv = 0;
    }*/
        float x1 = aGm[aGmStart], sigma;
        // Compute sigma first
        outputTensor = outQueue.AllocTensor<float>();
        if constexpr (isColumn)
        {
            // the frist loop
            //   Copy in
            copyInExtParams.blockCount = getQueueM();
            copyInExtParams.blockLen = sizeOfFloat;
            copyInExtParams.srcStride = (n_ - 1) * sizeOfFloat;
            copyInExtParams.dstStride = 0;

            inputTensor = inQueue.AllocTensor<float>();
            AscendC::DataCopyPad(inputTensor, aGm[aGmStart], copyInExtParams, copyInPadParams);
            inQueue.EnQue(inputTensor);

            // Compute
            inputTensor = inQueue.DeQue<float>();
            // copy the first inputTensor to outputTensor
            AscendC::DataCopy(outputTensor, inputTensor, getQueueM());
            AscendC::Duplicate(inputTensor, 0.0f, 1); // to calculate x[2:len]Tx[2:len],first put the first element to 0.0f
            AscendC::Mul(houseVec, inputTensor, inputTensor, getQueueM());
            AscendC::ReduceSum(inputTensor, houseVec, inputTensor[BlockFloatCnt], getQueueM());
            sigma = inputTensor(0);
            inQueue.FreeTensor(inputTensor);

            // formerLoopNum -1 loop
            for (int32_t i = 1; i < formerLoopNum; i++)
            {
                inputTensor = inQueue.AllocTensor<float>();
                AscendC::DataCopyPad(inputTensor, aGm[aGmStart + getQueueM() * i * n_], copyInExtParams, copyInPadParams);
                inQueue.EnQue(inputTensor);

                // Compute
                inputTensor = inQueue.DeQue<float>();
                AscendC::Mul(houseVec, inputTensor, inputTensor, getQueueM());
                AscendC::ReduceSum(inputTensor, houseVec, inputTensor[BlockFloatCnt], getQueueM());
                sigma += inputTensor(0);
                inQueue.FreeTensor(inputTensor);
            }

            // tailLoop
            if (tailBytes > 0)
            {
                copyInExtParams.blockCount = tailBytes / BlockSize;
                inputTensor = inQueue.AllocTensor<float>();
                AscendC::DataCopyPad(inputTensor, aGm[aGmStart + getQueueM() * formerLoopNum * n_], copyInExtParams, copyInPadParams);
                inQueue.EnQue(inputTensor);

                // Compute
                inputTensor = inQueue.DeQue<float>();
                AscendC::Mul(houseVec, inputTensor, inputTensor, copyInExtParams.blockCount);
                AscendC::ReduceSum(inputTensor, houseVec, inputTensor[BlockFloatCnt], copyInExtParams.blockCount);
                sigma += inputTensor(0);
                inQueue.FreeTensor(inputTensor);
            }
        }
        else
        {
            copyInExtParams.blockCount = 1;
            copyInExtParams.blockLen = FixedBufferSize;
            copyInExtParams.srcStride = 0;
            copyInExtParams.dstStride = 0;
            const uint32_t FixedBufferFloatCnt = FixedBufferSize / sizeOfFloat;
            // the first loop
            //  Copy in
            inputTensor = inQueue.AllocTensor<float>();
            AscendC::DataCopyPad(inputTensor, aGm[aGmStart], copyInExtParams, {});
            inQueue.EnQue(inputTensor);

            // Compute
            inputTensor = inQueue.DeQue<float>();
            // copy inputTensor to outputTensor
            AscendC::DataCopy(outputTensor, inputTensor, FixedBufferFloatCnt);
            AscendC::Duplicate(inputTensor, 0.0f, 1); // to calculate x[2:len]Tx[2:len],first put the first element to 0.0f
            AscendC::Mul(houseVec, inputTensor, inputTensor, FixedBufferFloatCnt);
            AscendC::ReduceSum(inputTensor, houseVec, inputTensor[BlockFloatCnt], FixedBufferFloatCnt);
            sigma = inputTensor(0);
            inQueue.FreeTensor(inputTensor);

            // formerLoopNum -1 loop
            for (int32_t i = 1; i < formerLoopNum; i++)
            {
                inputTensor = inQueue.AllocTensor<float>();
                AscendC::DataCopyPad(inputTensor, aGm[aGmStart + FixedBufferFloatCnt * i], copyInExtParams, {});
                inQueue.EnQue(inputTensor);
                // Compute
                inputTensor = inQueue.DeQue<float>();
                // copy inputTensor to outputTensor
                AscendC::DataCopy(outputTensor, inputTensor, FixedBufferFloatCnt);
                AscendC::Mul(houseVec, inputTensor, inputTensor, FixedBufferFloatCnt);
                AscendC::ReduceSum(inputTensor, houseVec, inputTensor[BlockFloatCnt], FixedBufferFloatCnt);
                sigma += inputTensor(0);
                inQueue.FreeTensor(inputTensor);
            }
            // tailLoop
            if (tailBytes > 0)
            {
                const uint32_t tailLoopLen = tailBytes / sizeOfFloat;
                copyInExtParams.blockLen = tailLoopLen + effectiveLen - ttl; // tailLoopLen-padLen
                inputTensor = inQueue.AllocTensor<float>();
                AscendC::DataCopyPad(inputTensor, aGm[aGmStart + FixedBufferFloatCnt * formerLoopNum], copyInExtParams, copyInPadParams);
                inQueue.EnQue(inputTensor);
                // Compute
                inputTensor = inQueue.DeQue<float>();
                AscendC::Mul(houseVec, inputTensor, inputTensor, tailLoopLen);
                AscendC::ReduceSum(inputTensor, houseVec, inputTensor[BlockFloatCnt], tailLoopLen);
                sigma += inputTensor(0);
                inQueue.FreeTensor(inputTensor);
            }
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
                copyOutExtParams.blockCount = getQueueM();
                copyOutExtParams.blockLen = sizeOfFloat;
                copyOutExtParams.srcStride = 0;
                copyOutExtParams.dstStride = (n_ - 1) * sizeOfFloat;
                copyInExtParams.blockCount = getQueueM();
                // firstLoop
                AscendC::Muls(outputTensor, outputTensor, v1_inv, getQueueM());
                AscendC::Duplicate(outputTensor, miu, 1);
                outQueue.EnQue(outputTensor);

                // Copy out
                outputTensor = outQueue.DeQue<float>();
                AscendC::DataCopyPad(aGm[aGmStart], outputTensor, copyOutExtParams);
                outQueue.FreeTensor(outputTensor);
                // formerLoopNum -1 loop
                for (int32_t i = 1; i < formerLoopNum; i++)
                {
                    // Copy In
                    inputTensor = inQueue.AllocTensor<float>();
                    AscendC::DataCopyPad(inputTensor, aGm[aGmStart + getQueueM() * i * n_], copyInExtParams, copyInPadParams);
                    inQueue.EnQue(inputTensor);
                    // Compute
                    inputTensor = inQueue.DeQue<float>();
                    outputTensor = outQueue.AllocTensor<float>();
                    AscendC::Muls(outputTensor, inputTensor, v1_inv, getQueueM());
                    outQueue.EnQue(outputTensor);

                    // Copy out
                    outputTensor = outQueue.DeQue<float>();
                    AscendC::DataCopyPad(aGm[aGmStart + getQueueM() * i * n_], outputTensor, copyOutExtParams);
                    outQueue.FreeTensor(outputTensor);
                }
                // tailLoop
                if (tailBytes > 0)
                {
                    copyOutExtParams.blockCount = tailBytes / BlockSize;
                    copyInExtParams.blockCount = tailBytes / BlockSize;
                    // Copy In
                    inputTensor = inQueue.AllocTensor<float>();
                    AscendC::DataCopyPad(inputTensor, aGm[aGmStart + getQueueM() * formerLoopNum * n_], copyInExtParams, copyInPadParams);
                    inQueue.EnQue(inputTensor);
                    // Compute
                    inputTensor = inQueue.DeQue<float>();
                    outputTensor = outQueue.AllocTensor<float>();
                    AscendC::Muls(outputTensor, inputTensor, v1_inv, copyInExtParams.blockCount);
                    outQueue.EnQue(outputTensor);

                    // Copy out
                    outputTensor = outQueue.DeQue<float>();
                    AscendC::DataCopyPad(aGm[aGmStart + getQueueM() * formerLoopNum * n_], outputTensor, copyOutExtParams);
                    outQueue.FreeTensor(outputTensor);
                }
            }
            else
            {
                copyOutExtParams.blockCount = 1;
                copyOutExtParams.blockLen = FixedBufferSize;
                copyOutExtParams.srcStride = 0;
                copyOutExtParams.dstStride = 0;
                copyInExtParams.blockLen = FixedBufferSize;
                // firstLoop
                // Copy In
                AscendC::Muls(outputTensor, outputTensor, v1_inv, FixedBufferFloatCnt);
                AscendC::Duplicate(outputTensor, miu, 1);
                outQueue.EnQue(outputTensor);

                // Copy out
                outputTensor = outQueue.DeQue<float>();
                AscendC::DataCopyPad(aGm[aGmStart], outputTensor, copyOutExtParams);
                outQueue.FreeTensor(outputTensor);

                // formerLoopNum -1 loop
                for (int32_t i = 1; i < formerLoopNum; i++)
                {
                    // Copy In
                    inputTensor = inQueue.AllocTensor<float>();
                    AscendC::DataCopyPad(inputTensor, aGm[aGmStart + FixedBufferFloatCnt * i], copyInExtParams, {});
                    inQueue.EnQue(inputTensor);
                    // Compute
                    inputTensor = inQueue.DeQue<float>();
                    outputTensor = outQueue.AllocTensor<float>();
                    AscendC::Muls(outputTensor, inputTensor, v1_inv, FixedBufferFloatCnt);
                    outQueue.EnQue(outputTensor);

                    // Copy out
                    outputTensor = outQueue.DeQue<float>();
                    AscendC::DataCopyPad(aGm[aGmStart + FixedBufferFloatCnt * i], outputTensor, copyOutExtParams);
                    outQueue.FreeTensor(outputTensor);
                }
                // tailLoop
                if (tailBytes > 0)
                {
                    const uint32_t tailLoopLen = tailBytes / sizeOfFloat;
                    copyOutExtParams.blockLen = tailLoopLen + effectiveLen - ttl; // tailLoopLen-padLen
                    copyInExtParams.blockLen = tailLoopLen + effectiveLen - ttl;
                    // Copy In
                    inputTensor = inQueue.AllocTensor<float>();
                    AscendC::DataCopyPad(inputTensor, aGm[aGmStart + FixedBufferFloatCnt * formerLoopNum], copyInExtParams, copyInPadParams);
                    inQueue.EnQue(inputTensor);
                    // Compute
                    inputTensor = inQueue.DeQue<float>();
                    outputTensor = outQueue.AllocTensor<float>();
                    AscendC::Muls(outputTensor, inputTensor, v1_inv, tailLoopLen);
                    outQueue.EnQue(outputTensor);

                    // Copy out
                    outputTensor = outQueue.DeQue<float>();
                    AscendC::DataCopyPad(aGm[aGmStart + FixedBufferFloatCnt * formerLoopNum], outputTensor, copyOutExtParams);
                    outQueue.FreeTensor(outputTensor);
                }
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
            for (int32_t j = jFirst; j < n_; j += aivNum)
            {
                LoadHouseVecAndApplyTransformCoreTiling(ttl, i * n_ + j, colPadParams, copyInExtParams, copyInExtParams, copyOutExtParams, beta, aGm);
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
            for (int32_t j = jFirst; j < m_; j += aivNum)
            {
                LoadHouseVecAndApplyTransformCoreTiling(ttl, j * n_ + i + 1, rowPadParams, copyInExtParams, copyInExtParams, copyOutExtParams, beta, aGm);
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

    __aicore__ inline void LoadHouseVecAndApplyTransformCoreTiling(
        const uint32_t ttl,
        const uint32_t targetGmStart,
        const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
        const AscendC::DataCopyExtParams &copyInHouseVecExtParams,
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
                    for (int32_t j = jFirst; j < m_; j += aivNum)
                    {
                        LoadHouseVecAndApplyTransformCoreTiling(ttl, i * m_ + j, colPadParams, copyInExtParams, copyInExtParams, copyOutExtParams, beta, uGm);
                    }
                }
                AscendC::SyncAll<true>(gmWorkspace, ubWorkspace);
            }

            // get Vt
            for (int32_t i = n_ - 2; i >= 0; i--)
            {
                // the i-th householder vector updates n_-i rows,n-i-1 columns of Vt
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
                        for (int32_t j = jFirst; j < n_; j += aivNum)
                        {
                            LoadHouseVecAndApplyTransformCoreTiling(ttl, j * n_ + i + 1, rowPadParams, copyInExtParams, copyInExtParams, copyOutExtParams, beta, vtGm);
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
#ifndef _____PIPE_INSIDECLASS
        AscendC::TPipe pipe;
        if (auto UBsizeRequired = notTilingKGKBSize(M, N); UBsizeRequired < (192 << 10))
        {
            if (AscendC::GetBlockIdx() == 0)
            {
                AscendC::printf("the UBsizeRequired is %d\n", UBsizeRequired);
            }
            Kernel_Golub_Kahan_Bidiagonalization kernel;
            kernel.Init(M, N, a, u, vt, d, e, tauq, taup, workspace, pipe);
#else
        Kernel_Golub_Kahan_Bidiagonalization kernel;
        kernel.Init(M, N, a, u, vt, d, e, tauq, taup, workspace);
#endif
            kernel.Process();
        }
        else
        {
#ifndef _____PIPE_INSIDECLASS
            Kernel_Golub_Kahan_Bidiagonalization<true> kernel;
            kernel.Init(M, N, a, u, vt, d, e, tauq, taup, workspace, pipe);
#else
        Kernel_Golub_Kahan_Bidiagonalization<true> kernel;
        kernel.Init(M, N, a, u, vt, d, e, tauq, taup, workspace);
#endif
            kernel.Process();
        }
    }
