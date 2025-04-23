#include "kernel_operator.h"
#include <cmath>
#include "kernel_log.h"
constexpr uint32_t sizeOfFloat = sizeof(float);
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t maxL1FloatSize = 1 << (19 - 2);
constexpr int32_t BlockSize = 32;
constexpr int32_t CONCURRENT_COL_CNT = BlockSize / sizeOfFloat;
constexpr int32_t SizePerOperation = 256;
constexpr int32_t BlockNumPerOperation = SizePerOperation / BlockSize;
// constexpr uint64_t MASK_PATTERN = 0x0101010101010101ULL;
const AscendC::DataCopyPadExtParams<float> colPadParams = {true, 0, 7, 0.0f};
// #define _____PIPE_INSIDECLASS
template <bool ifTiling = false>
class Kernel_Golub_Kahan_Bidiagonalization
{
public:
    __aicore__ inline Kernel_Golub_Kahan_Bidiagonalization() {}
#ifdef _____PIPE_INSIDECLASS
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR u, GM_ADDR vt, int M, int N, GM_ADDR d, GM_ADDR e)
#else
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR u, GM_ADDR vt, int M, int N, GM_ADDR d, GM_ADDR e, AscendC::TPipe &pipe)
#endif
    {
        if (AscendC::GetBlockIdx() != 0)
            return;
        m_ = M;
        n_ = N;
        ASSERT(M >= N && "in Init, we should have M>=N");
        // k_ = std::min(M, N);

        // 设置全局内存缓冲区
        aGm.SetGlobalBuffer((__gm__ float *)a, M * N);
        uGm.SetGlobalBuffer((__gm__ float *)u, M * M);
        vtGm.SetGlobalBuffer((__gm__ float *)vt, N * N);
        dGm.SetGlobalBuffer((__gm__ float *)d, N);
        eGm.SetGlobalBuffer((__gm__ float *)e, N - 1);

        initUV();

        // 初始化管道缓冲区
        // pipe.InitBuffer(colQueue, BUFFER_NUM, M * BlockSize + SizePerOperation);
        // pipe.InitBuffer(rowQueue, BUFFER_NUM, N * sizeOfFloat);
        // pipe.InitBuffer(outQueue, BUFFER_NUM, M * BlockSize + SizePerOperation);
        pipe.InitBuffer(inQueue, BUFFER_NUM, M * BlockSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, M * BlockSize);
        pipe.InitBuffer(d_, N * sizeOfFloat);
        pipe.InitBuffer(e_, (N - 1) * sizeOfFloat);
        pipe.InitBuffer(tauq_, N * sizeOfFloat);
        pipe.InitBuffer(taup_, (N - 1) * sizeOfFloat);
        pipe.InitBuffer(householder_vec, M * BlockSize);
        pipe.InitBuffer(scalarTmp, sizeOfFloat);
        pipe.InitBuffer(vecTmp, M * BlockSize);
        // pipe.InitBuffer(householder_vec, M * BlockSize + SizePerOperation); // an additional SizePerOperation in case tail masked operation exceeds the address boundary
        // reduce sum need at most M
        // pipe.InitBuffer(worktmp, M * BlockSize + SizePerOperation);

        houseVec = householder_vec.Get<float>();
        scalartmp = scalarTmp.Get<float>();
        vectmp = vecTmp.Get<float>();
        taup = taup_.Get<float>();
        tauq = tauq_.Get<float>();
    }
    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() != 0)
            return;
        // 主循环：对每一列和行进行Householder变换
        for (int32_t i = 0; i < n_; i++)
        {
            // AscendC::printf("ith column transform: %d\n", i);
            if (i < n_)
            {
                // 计算列的Householder变换
                ComputeColumnHouseholderV2(i);
                // 应用列变换到剩余的矩阵
                ApplyColumnTransform(i);
            }

            // AscendC::printf("ith row transform: %d\n", i);
            if (i < n_ - 1)
            {
                // 计算行的Householder变换
                ComputeRowHouseholderV2(i);
                // 应用行变换到剩余的矩阵
                ApplyRowTransform(i);
            }
        }

        GetUVt();
    }

private:
    __aicore__ inline void ComputeColumnHouseholderV2(int32_t i)
    {
        ASSERT(i < n_ && "in ComputeColumnHouseholder, i should be less than n_");
        const uint16_t len = m_ - i;
        if (len <= 1)
        {
            // a scalar vector, no need to calc,beta=0;
            tauq(i) = 0;
            dGm(i) = aGm(i * n_ + i);
            return;
        }
        const uint32_t ttl = len * CONCURRENT_COL_CNT, aGmStart = i * n_ + i;
        const AscendC::DataCopyExtParams copyInExtParams = {len, 4, (n_ - 1) * sizeOfFloat, 0, 0};
        const AscendC::DataCopyExtParams copyOutExtParams = {len, 4, 0, (n_ - 1) * sizeOfFloat, 0};
        auto tau = tauq[i];
        auto deGm = dGm[i];
        ComputeHouseholder(ttl, aGmStart, colPadParams, copyInExtParams, copyOutExtParams, tau, deGm);
    }
    __aicore__ inline void ComputeRowHouseholderV2(int32_t i)
    {
        ASSERT(i < n_ - 1 && "in ComputeRowHouseholder, i should be less than n_ - 1");
        const uint16_t len = n_ - i - 1;
        const uint8_t padLen = len % 8 == 0 ? 0 : 8 - len % 8;
        const uint32_t ttl = len + padLen;
        const AscendC::DataCopyPadExtParams<float> rowPadParams = {true, 0, padLen, 0.0f};
        const AscendC::DataCopyExtParams copyInExtParams = {1, len * sizeOfFloat, 0, 0, 0};
        const AscendC::DataCopyExtParams copyOutExtParams = {1, len * sizeOfFloat, 0, 0, 0};
        auto tau = taup[i];
        auto deGm = eGm[i];
        ComputeHouseholder(ttl, i * n_ + i + 1, rowPadParams, copyInExtParams, copyOutExtParams, tau, deGm);
    }
    __aicore__ inline void ComputeHouseholder(const uint32_t ttl, const uint32_t aGmStart,
                                              const AscendC::DataCopyPadExtParams<float> &copyInPadParams,
                                              const AscendC::DataCopyExtParams &copyInExtParams,
                                              const AscendC::DataCopyExtParams &copyOutExtParams,
                                              AscendC::LocalTensor<float> &tau,
                                              AscendC::GlobalTensor<float> &deGm)
    {
        // Copy in
        inputTensor = inQueue.AllocTensor<float>();
        AscendC::DataCopyPad(inputTensor, aGm[aGmStart], copyInExtParams, copyInPadParams);
        inQueue.EnQue(inputTensor);

        // Vector Compute
        inputTensor = inQueue.DeQue<float>();
        outputTensor = outQueue.AllocTensor<float>();
        // copy inputTensor to outputTensor
        AscendC::Adds(outputTensor, inputTensor, 0.0f, ttl);
        auto x1 = inputTensor(0);
        AscendC::Duplicate(inputTensor, 0.0f, 1); // to calculate x[2:len]Tx[2:len],first put the first element to 0.0f
        AscendC::Mul(houseVec, inputTensor, inputTensor, ttl);
        AscendC::ReduceSum(inputTensor, houseVec, inputTensor[CONCURRENT_COL_CNT], ttl);
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
            float v1;
            if (x1 <= 0)
            {
                v1 = x1 - miu;
            }
            else
            {
                v1 = -sigma / (x1 + miu);
            }
            float v1sq = v1 * v1;
            tau(0) = 2 * v1sq / (sigma + v1sq);
            AscendC::Muls(outputTensor, outputTensor, 1.0f / v1, ttl);
            AscendC::Adds(houseVec, outputTensor, 0.0f, ttl);
            AscendC::Duplicate(houseVec, 1.0f, 1);
            AscendC::Duplicate(outputTensor, miu, 1);
            deGm(0) = miu;
            outQueue.EnQue(outputTensor);

            // Copy out
            outputTensor = outQueue.DeQue<float>();
            AscendC::DataCopyPad(aGm[aGmStart], outputTensor, copyOutExtParams);
            outQueue.FreeTensor(outputTensor);
        }
    }
    __aicore__ inline void ApplyColumnTransform(int32_t i)
    {

        // 应用列变换到右侧子矩阵
        const uint16_t len = m_ - i;
        float beta = tauq(i);
        if (beta == 0)
        {
            return;
        }
        for (int32_t j = i + 1; j < n_; j++)
        {

            // data loaded in col
            col = inQueue.AllocTensor<float>();
            const AscendC::DataCopyExtParams dataCopyExtParams = {len, 4, (n_ - 1) * sizeOfFloat, 0, 0};
            AscendC::DataCopyPad(col, aGm[i * n_ + j], dataCopyExtParams, colPadParams);
            inQueue.EnQue(col);
            // DumpTensor(col,5, 6 * CONCURRENT_COL_CNT);
            // AscendC::printf("in ApplyColumnTransform,i=%d,j=%d\n",i,j);
            ApplyTransformCore(len * CONCURRENT_COL_CNT, beta);

            // copy out
            AscendC::LocalTensor<float> outcol = outQueue.DeQue<float>();
            const AscendC::DataCopyExtParams dataCopyExtParams2 = {len, 4, 0, (n_ - 1) * sizeOfFloat, 0};
            AscendC::DataCopyPad(aGm[i * n_ + j], outcol, dataCopyExtParams2);
            outQueue.FreeTensor(outcol);
        }
    }
    __aicore__ inline void ApplyRowTransform(int32_t i)
    {
        // 应用行变换到下方子矩阵A[i+1:m,i+1:n]
        const uint16_t len = n_ - i - 1;
        const uint8_t padLen = len % 8 == 0 ? 0 : 8 - len % 8;
        const uint32_t ttl = len + padLen;
        float beta = taup(i);
        if (beta == 0)
        {
            return;
        }
        for (int32_t j = i + 1; j < m_; j++)
        {
            // 加载当前行
            row = inQueue.AllocTensor<float>();
            AscendC::DataCopyPadExtParams<float> rowPadParams = {true, 0, padLen, 0.0f};
            const AscendC::DataCopyExtParams dataCopyExtParams = {1, len * sizeOfFloat, 0, 0, 0};
            AscendC::DataCopyPad(row, aGm[j * n_ + i + 1], dataCopyExtParams, rowPadParams);
            inQueue.EnQue(row);

            ApplyTransformCore(ttl, beta);

            // copy out
            AscendC::LocalTensor<float> outrow = outQueue.DeQue<float>();
            const AscendC::DataCopyExtParams dataCopyExtParams2 = {1, len * sizeOfFloat, 0, 0, 0};
            AscendC::DataCopyPad(aGm[j * n_ + i + 1], outrow, dataCopyExtParams2);
            outQueue.FreeTensor(outrow);
        }
    }

    __aicore__ inline void ApplyTransformCore(const int32_t ttl, const float beta)
    {
        // the col compute is the same as the row compute
        //  Compute
        row = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outrow = outQueue.AllocTensor<float>();
        // row*v
        AscendC::Mul(vectmp, row, houseVec, ttl);
        AscendC::ReduceSum(scalartmp, vectmp, outrow, ttl);
        // beta*row*v
        float coeff = scalartmp(0);
        coeff *= beta;
        AscendC::Muls(vectmp, houseVec, coeff, ttl);
        // row - beta*row*v*vT
        AscendC::Sub(outrow, row, vectmp, ttl);
        // AscendC::printf("outrow in ApplyTransformCore:\n");
        // DumpTensor(outrow,5, 6 * CONCURRENT_COL_CNT);

        inQueue.FreeTensor(row);
        outQueue.EnQue(outrow);
    }
    __aicore__ inline void GetUVt()
    {
        // print uGm
        //  {
        //      for(int32_t i=0;i<m_;i++)
        //      {
        //          for(int32_t j=0;j<m_;j++)
        //          {
        //              AscendC::printf("uGm[%d][%d]=%f\n",i,j,uGm(i * m_ + j));
        //          }
        //      }
        //  }
        // 刷新Cache，保证uGm、vGm与Cache的一致性
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(uGm);
        // get U
        for (int32_t i = n_ - 1; i >= 0; i--)
        {
            // the i-th householder vector updates m_-i columns of U
            const uint16_t len = m_ - i;
            auto beta = tauq(i);
            if (beta == 0)
            {
                continue;
            }

            // 加载当前houseVec
            col = inQueue.AllocTensor<float>();
            const AscendC::DataCopyExtParams dataCopyExtParams = {len, 4, (n_ - 1) * sizeOfFloat, 0, 0};
            AscendC::DataCopyPad(col, aGm[i * n_ + i], dataCopyExtParams, colPadParams);
            inQueue.EnQue(col);
            col = inQueue.DeQue<float>();
            AscendC::Adds(houseVec, col, .0f, len * CONCURRENT_COL_CNT);
            houseVec(0) = 1.0f;
            // DumpTensor(houseVec,5, 6 * CONCURRENT_COL_CNT);
            inQueue.FreeTensor(col);

            for (int32_t j = i; j < m_; j++)
            {
                col = inQueue.AllocTensor<float>();
                const AscendC::DataCopyExtParams dataCopyExtParams = {len, 4, (m_ - 1) * sizeOfFloat, 0, 0};
                AscendC::DataCopyPad(col, uGm[i * m_ + j], dataCopyExtParams, colPadParams);
                // AscendC::printf("col in GetUVt: ,i=%d,j=%d\n",i,j);
                // DumpTensor(col, 5, 6 * CONCURRENT_COL_CNT);
                inQueue.EnQue(col);

                ApplyTransformCore(len * CONCURRENT_COL_CNT, beta);

                // copy out
                AscendC::LocalTensor<float> outcol = outQueue.DeQue<float>();
                // AscendC::printf("outcol in GetUVt: ,i=%d,j=%d\n",i,j);
                // DumpTensor(outcol, 5, 6 * CONCURRENT_COL_CNT);

                const AscendC::DataCopyExtParams dataCopyExtParams2 = {len, 4, 0, (m_ - 1) * sizeOfFloat, 0};
                AscendC::DataCopyPad(uGm[i * m_ + j], outcol, dataCopyExtParams2);

                outQueue.FreeTensor(outcol);
            }
        }
        // print uGm
        // {
        //     for(int32_t i=0;i<m_;i++)
        //     {
        //         for(int32_t j=0;j<m_;j++)
        //         {
        //             AscendC::printf("uGm[%d][%d]=%f\n",i,j,uGm(i * m_ + j));
        //         }
        //     }
        // }
        // get Vt
        for (int32_t i = n_ - 2; i >= 0; i--)
        {
            // the i-th householder vector updates n_-i rows,n-i-1 columns of Vt
            const uint16_t len = n_ - i - 1;
            const uint8_t padLen = len % 8 == 0 ? 0 : 8 - len % 8;
            const uint32_t ttl = len + padLen;
            auto beta = taup(i);
            if (beta == 0)
            {
                continue;
            }

            // 加载当前houseVec
            row = inQueue.AllocTensor<float>();
            AscendC::DataCopyPadExtParams<float> rowPadParams = {true, 0, padLen, 0.0f};
            const AscendC::DataCopyExtParams dataCopyExtParams = {1, len * sizeOfFloat, 0, 0, 0};
            AscendC::DataCopyPad(row, aGm[i * n_ + i + 1], dataCopyExtParams, rowPadParams);
            inQueue.EnQue(row);
            row = inQueue.DeQue<float>();
            AscendC::Adds(houseVec, row, .0f, ttl);
            houseVec(0) = 1.0f;
            inQueue.FreeTensor(row);

            for (int32_t j = i; j < n_; j++)
            {
                row = inQueue.AllocTensor<float>();
                const AscendC::DataCopyExtParams dataCopyExtParams = {1, len * sizeOfFloat, 0, 0, 0};
                AscendC::DataCopyPad(row, vtGm[j * n_ + i + 1], dataCopyExtParams, rowPadParams);
                inQueue.EnQue(row);

                ApplyTransformCore(ttl, beta);

                // copy out
                AscendC::LocalTensor<float> outrow = outQueue.DeQue<float>();
                const AscendC::DataCopyExtParams dataCopyExtParams2 = {1, len * sizeOfFloat, 0, 0, 0};
                AscendC::DataCopyPad(vtGm[j * n_ + i + 1], outrow, dataCopyExtParams2);
                outQueue.FreeTensor(outrow);
            }
        }
    }
    __aicore__ inline void initUV()
    {
        // cache 问题，AIV处理前需刷新Cache
        for (int32_t i = 0; i < m_; i++)
        {
            uGm(i * m_ + i) = 1.0f;
        }
        for (int32_t i = 0; i < n_; i++)
        {
            vtGm(i * n_ + i) = 1.0f;
        }
    }

private:
    uint16_t m_, n_;
    // int32_t k_;
#ifdef _____PIPE_INSIDECLASS
    AscendC::TPipe pipe;
#endif
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    // AscendC::TBuf<AscendC::TPosition::VECCALC> colBuf, rowBuf, outBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> d_, e_, tauq_, taup_, householder_vec, scalarTmp, vecTmp;
    // AscendC::TBuf<AscendC::TPosition::VECCALC> worktmp;
    AscendC::GlobalTensor<float> aGm, uGm, vtGm, dGm, eGm;
    AscendC::LocalTensor<float> col, row, houseVec, scalartmp, vectmp, taup, tauq, inputTensor, outputTensor;
};

template <>
class Kernel_Golub_Kahan_Bidiagonalization<true>
{
};
extern "C" __global__ __aicore__ void upper_bidiagonalization(GM_ADDR a, GM_ADDR u, GM_ADDR vt, int M, int N, GM_ADDR d, GM_ADDR e)
{
#ifndef _____PIPE_INSIDECLASS
    AscendC::TPipe pipe;
    auto blockNum = AscendC::GetBlockNum();
    Kernel_Golub_Kahan_Bidiagonalization kernel;
    kernel.Init(a, u, vt, M, N, d, e, pipe);
#else
    Kernel_Golub_Kahan_Bidiagonalization kernel;
    kernel.Init(a, u, vt, M, N, d, e);
#endif
    kernel.Process();
}
