#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t maxL1FloatSize = 1 << 19 - 2;
constexpr int32_t BlockSize = 32;
constexpr int32_t CONCURRENT_COL_CNT = BlockSize / sizeof(float);
constexpr int32_t SizePerOperation = 256;
constexpr int32_t BlockNumPerOperation = SizePerOperation / BlockSize;
constexpr uint64_t MASK_PATTERN = 0x0101010101010101ULL;
class Kernel_Golub_Kahan_Bidiagonalization
{
public:
    __aicore__ inline Kernel_Golub_Kahan_Bidiagonalization() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR u, GM_ADDR vt, int M, int N)
    {
        m_ = M;
        n_ = N;
        assert(M >= N && "in Init, we should have M>=N");
        k_ = std::min(M, N);
        initUV(u, vt);

        // 设置全局内存缓冲区
        aGm.SetGlobalBuffer((__gm__ float *)a, M * N);
        uGm.SetGlobalBuffer((__gm__ float *)u, M * M);
        vtGm.SetGlobalBuffer((__gm__ float *)vt, N * N);

        // 初始化管道缓冲区
        // pipe.InitBuffer(colQueue, BUFFER_NUM, M * BlockSize + SizePerOperation);
        // pipe.InitBuffer(rowQueue, BUFFER_NUM, N * sizeof(float));
        // pipe.InitBuffer(outQueue, BUFFER_NUM, M * BlockSize + SizePerOperation);
        pipe.InitBuffer(inQueue, BUFFER_NUM, M * BlockSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, M * BlockSize);
        pipe.InitBuffer(d_, N * sizeof(float));
        pipe.InitBuffer(e_, (N - 1) * sizeof(float));
        pipe.InitBuffer(tauq_, N * sizeof(float));
        pipe.InitBuffer(taup_, (N - 1) * sizeof(float));
        pipe.InitBuffer(householder_vec, M * BlockSize);
        pipe.InitBuffer(scalarTmp, sizeof(float));
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
        // 主循环：对每一列和行进行Householder变换
        for (int32_t i = 0; i < k_; i++)
        {
            if (i < n_)
            {
                // 计算列的Householder变换
                ComputeColumnHouseholder(i);
                // 应用列变换到剩余的矩阵
                ApplyColumnTransform(i);
            }

            if (i < n_ - 1)
            {
                // 计算行的Householder变换
                ComputeRowHouseholder(i);
                // 应用行变换到剩余的矩阵
                ApplyRowTransform(i);
            }
        }

        GetUVt();
    }

private:
    /*     __aicore__ inline void ComputeColumnHouseholder(int32_t i)
        {
            assert(i < n_ && "in ComputeColumnHouseholder, i should be less than n_");
            const int32_t len = _m - i;
            if (len <= 1)
            {
                // a scalar vector, no need to calc,beta=0;
                tauq_(i) = 0;
                return;
            }
            // col= colQueue.AllocTensor<float>();
            col = colBuf.Get<float>();
            const auto floor_i = (i / CONCURRENT_COL_CNT) * CONCURRENT_COL_CNT, offset_i = i % CONCURRENT_COL_CNT;
            uint64_t mask[1] = {MASK_PATTERN << offset_i};
            // const uint8_t repeatTimes = (len - 1) / BlockNumPerOperation, tailSize = (len - 1) % BlockNumPerOperation;
            // const uint64_t tailOffset = repeatTimes * SizePerOperation + CONCURRENT_COL_CNT;
            // if tailSize > 0,only tailSize个 bit is 1
            // uint64_t tailMask[1] = {MASK_PATTERN << (CONCURRENT_COL_CNT * (BlockNumPerOperation - tailSize))};
            // 加载当前列,floor_i 列的block，加载M-i块
            AscendC::DataCopy(col, aGm[floor_i * n_], {len, 1, n_ / CONCURRENT_COL_CNT, 0});
            // colQueue.EnQue(col);

            // Vector Compute
            // col = colQueue.DeQue<float>();
            // AscendC::LocalTensor<float> outcol = outQueue.AllocTensor<float>();
            AscendC::LocalTensor<float> outcol = outBuf.Get<float>();
            AscendC::LocalTensor<float> tmp = householder_vec.Get<float>();
            // pad the tail operation blocks with 0
            AscendC::Duplicate(col[len * CONCURRENT_COL_CNT], .0f, BlockNumPerOperation * CONCURRENT_COL_CNT);
            // copy col to outcol
            AscendC::Adds(outcol, col, .0f, len * CONCURRENT_COL_CNT);
            // get the first element of x
            auto x1 = col(offset_i);
            // calculate the 2-norm of x except the first element
            // round up repeatTimes to deal the tail together
            AscendC::Mul(tmp, col[CONCURRENT_COL_CNT], col[CONCURRENT_COL_CNT], mask, (len - 1 + BlockNumPerOperation - 1) / BlockNumPerOperation, {1, 1, 1, BlockNumPerOperation, BlockNumPerOperation, BlockNumPerOperation});
            // use col as tmp worklocal space,col[0]=x(2:)Tx(2:)
            AscendC::ReduceSum(col, tmp, col[CONCURRENT_COL_CNT], mask, (len - 1 + BlockNumPerOperation - 1) / BlockNumPerOperation, BlockNumPerOperation);
            // update tauq_[i] as beta,
            auto sigma = col(0);
            colQueue.FreeTensor(col);
            if (sigma == 0)
            {
                // zero but x1
                if (x1 >= 0)
                {
                    tauq_(i) = 0;
                    // since beta is 0,no need to update other columns
                    // and nothing changes to the A matrix
                }
                else
                {
                    tauq_(i) = -2;
                    AscendC::Duplicate(tmp, .0f, len * CONCURRENT_COL_CNT);
                    // v1=1.0f
                    tmp(offset_i) = 1.0f;
                    // in column i only changes the A[i][i] to its negative,which is -outcol(offset_i)
                    aGm[i * n_ + i] = -outcol(offset_i);
                    // TODO check whether this practice is fine
                    //  outQueue.FreeTensor(outcol);
                }
            }
            else
            {
                auto miu = std::sqrt(x1 * x1 + sigma);
                float v1;
                if (x1 <= 0)
                { // update v1,aka,tmp(offset_i)
                    v1 = x1 - miu;
                }
                else
                {
                    v1 = -sigma / (x1 + miu);
                }
                float v1sq = v1 * v1;
                tauq_(i) = 2 * v1sq / (sigma + v1sq);
                // calculate the essential householder vector
                AscendC::Muls(outcol, outcol, 1.0f / v1, mask, (len + BlockNumPerOperation - 1) / BlockNumPerOperation, {1, 1, BlockNumPerOperation, BlockNumPerOperation});
                // construct the householder vector v,use adds 0 to copy conveniently
                AscendC::Adds(tmp, outcol, .0f, len * CONCURRENT_COL_CNT);
                // update final v1 and final A[i][i] using tmp and outcol
                tmp(offset_i) = 1.0f;
                outcol(offset_i) = miu; // 2-norm of x
                // outQueue.Enque(outcol);

                // outcol=outQueue.Deque<float>();
                AscendC::DataCopy(aGm[floor_i * n_], outcol, {len, 1, 0, n_ / CONCURRENT_COL_CNT});
            }
        }
        */
    __aicore__ inline void ComputeColumnHouseholder(int32_t i)
    {
        assert(i < n_ && "in ComputeColumnHouseholder, i should be less than n_");
        const int32_t len = m_ - i;
        if (len <= 1)
        {
            // a scalar vector, no need to calc,beta=0;
            tauq(i) = 0;
            return;
        }

        // 加载当前i列的共len个数据,use 0.0f to pad right
        col = inQueue.AllocTensor<float>();
        // use 0.0f to pad right
        AscendC::DataCopyPad(col, aGm[i * n_ + i], {len, 4, n_ * sizeof(float), 0}, {true, 0, 7, 0.0f});
        inQueue.EnQue(col);

        // Vector Compute
        col = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outcol = outQueue.AllocTensor<float>();
        // copy col to outcol
        AscendC::Adds(outcol, col, .0f, len * CONCURRENT_COL_CNT);
        // get the first element of x
        auto x1 = col(0);
        // calculate the 2-norm of x except the first element
        AscendC::Mul(houseVec, col[CONCURRENT_COL_CNT], col[CONCURRENT_COL_CNT], (len - 1) * CONCURRENT_COL_CNT);
        // use col as tmp worklocal space,col[0]=x(2:)Tx(2:)
        // since len >1,so col[CONCURRENT_COL_CNT] have more blocks than required worklocal
        AscendC::ReduceSum(col, houseVec, col[CONCURRENT_COL_CNT],(len-1)*CONCURRENT_COL_CNT));
        // update tauq_[i] as beta,
        auto sigma = col(0);
        inQueue.FreeTensor(col);
        if (sigma == 0)
        {
            // x[2:m] are all zero
            tauq(i) = 0;
            // since beta is 0,no need to update other columns
            // and nothing changes to the A matrix
            // TODO check whether this practice is fine
            outQueue.FreeTensor(outcol);
        }
        else
        {
            auto miu = std::sqrt(x1 * x1 + sigma);
            float v1;
            if (x1 <= 0)
            { // update v1,aka,houseVec(0)
                v1 = x1 - miu;
            }
            else
            {
                v1 = -sigma / (x1 + miu);
            }
            float v1sq = v1 * v1;
            tauq(i) = 2 * v1sq / (sigma + v1sq);
            // calculate the essential householder vector
            AscendC::Muls(outcol, outcol, 1.0f / v1, len * CONCURRENT_COL_CNT);
            // construct the householder vector v,use adds 0 to copy conveniently
            AscendC::Adds(houseVec, outcol, .0f, len * CONCURRENT_COL_CNT);
            // update final v1 and final A[i][i] using tmp and outcol
            houseVec(0) = 1.0f;
            outcol(0) = miu; // 2-norm of x
            outQueue.Enque(outcol);

            outcol = outQueue.Deque<float>();
            AscendC::DataCopyPad(aGm[i * n_ + i], outcol, {len, 4, 0, n_ * sizoef(float)});
            outQueue.FreeTensor(outcol);
        }
    }
    __aicore__ inline void ComputeRowHouseholder(int32_t i)
    {
        assert(i < n_ - 1 && "in ComputeRowHouseholder, i should be less than n_ - 1");
        const int32_t len = n_ - i - 1;
        const int32_t padLen = len % 8 == 0 ? 0 : 8 - len % 8, ttl = len + padLen;
        auto taup = taup_.Get<float>();
        if (len <= 1)
        {
            taup(i) = 0;
            return;
        }
        // Copy in
        row = inQueue.AllocTensor<float>();
        AscendC::DataCopyPad(row, aGm[i * n_ + i + 1], {1, len * 4, 0, 0}, {true, 0, padLen, 0.0f});
        inQueue.EnQue(row);

        // Vector Compute
        row = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outrow = outQueue.AllocTensor<float>();
        // copy row to outrow
        AscendC::Adds(outrow, row, 0.0f, ttl);
        auto x1 = row(0);
        row(0) = 0.0f; // to calculate x[2:len]Tx[2:len]
        AscendC::Mul(houseVec, row, row, ttl);
        AscendC::ReduceSum(row, houseVec, row[CONCURRENT_COL_CNT], ttl);
        auto sigma = row(0);
        inQueue.FreeTensor(row);
        if (sigma == 0)
        {
            taup(i) = 0;
            outQueue.FreeTensor(outrow);
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
            taup(i) = 2 * v1sq / (sigma + v1sq);
            AscendC::Muls(outrow, outrow, 1.0f / v1, ttl);
            AscendC::Adds(houseVec, outrow, 0.0f, ttl);
            houseVec(0) = 1.0f;
            outrow(0) = miu;
            outQueue.Enque(outrow);

            outrow = outQueue.Deque<float>();
            AscendC::DataCopyPad(aGm[i * n_ + i + 1], outrow, {1, len * 4, 0, 0});
            outQueue.FreeTensor(outrow);
        }
    }
    __aicore__ inline void ApplyColumnTransform(int32_t i)
    {

        // 应用列变换到右侧子矩阵
        const int32_t len = _m - i;
        float beta = tauq(i);
        if (beta == 0)
        {
            return;
        }
        for (int32_t j = i + 1; j < n_; j++)
        {

            // data loaded in col
            col = inQueue.AllocTensor<float>();
            AscendC::DataCopyPad(col, aGm[i * n_ + j], {len, 4, n_ * sizeof(float), 0}, {true, 0, 7, 0.0f});
            inQueue.EnQue(col);

            ApplyTransformCore(len*CONCURRENT_COL_CNT,beta);

            // copy out
            AscendC::LocalTensor<float> outcol = outQueue.Deque<float>();
            AscendC::DataCopyPad(aGm[i * n_ + j], outcol, {len, 4, 0, n_ * sizeof(float)});
            outQueue.FreeTensor(outcol);
        }
    }
    __aicore__ inline void ApplyRowTransform(int32_t i)
    {
        // 应用行变换到下方子矩阵A[i+1:m,i+1:n]
        const int32_t len = n_ - i - 1;
        const int32_t padLen = len % 8 == 0 ? 0 : 8 - len % 8, ttl = len + padLen;
        float beta = taup(i);
        if (beta == 0)
        {
            return;
        }
        for (int32_t j = i + 1; j < m_; j++)
        {
            // 加载当前行
            row = inQueue.AllocTensor<float>();
            AscendC::DataCopyPad(row, aGm[j * n_ + i + 1], {1, len * 4, 0, 0}, {true, 0, padLen, 0.0f});
            inQueue.EnQue(row);

            ApplyTransformCore(ttl,beta);

            // copy out
            AscendC::LocalTensor<float> outrow = outQueue.Deque<float>();
            AscendC::DataCopyPad(aGm[j * n_ + i + 1], outrow, {1, len * 4, 0, 0});
            outQueue.FreeTensor(outrow);
        }
    }

    __aicore__ inline void ApplyTransformCore(const int32_t ttl,const float beta)
    {
            //the col compute is the same as the row compute
            // Compute
            row = inQueue.DeQue<float>();
            AscendC::LocalTensor<float> outrow = outQueue.AllocTensor<float>();
            // row*v
            AscendC::Mul(vectmp, row, houseVec, ttl);
            AscendC::ReduceSum(scalartmp, vectmp, outrow, ttl);
            // beta*row*v
            AscendC::Muls(scalartmp, scalartmp, beta, 1);
            AscendC::Muls(vectmp, houseVec, scalartmp(0), ttl);
            // row - beta*row*v*vT
            Ascend::Sub(outrow, row, vectmp, ttl);
            inQueue.FreeTensor(row);
            outQueue.Enque(outrow);
    }
    __aicore__ inline void GetUVt()
    {
        //get U
        for(int32_t i=n_-1;i>=0;i--)
        {
            //the i-th householder vector updates m_-i columns of U
            const int32_t len=m_-i;
            auto beta=tauq(i);
            if(beta==0)
            {
                continue;
            }


            // 加载当前houseVec
            col = inQueue.AllocTensor<float>();
            AscendC::DataCopyPad(col, aGm[i * n_ + i], {len, 4, n_ * sizeof(float), 0}, {true, 0, 7, 0.0f});
            inQueue.EnQue(col);
            col = inQueue.DeQue<float>();
            AscendC::Adds(houseVec, col, .0f, len*CONCURRENT_COL_CNT);
            houseVec(0)=1.0f;
            inQueue.FreeTensor(col);
            
            for (int32_t j = i; j < m_; j++)
            {
                col=inQueue.AllocTensor<float>();
                AscendC::DataCopyPad(col, uGm[i * m_ + j], {len, 4, m_ * sizeof(float), 0}, {true, 0, 7, 0.0f});
                inQueue.EnQue(col);

                ApplyTransformCore(len*CONCURRENT_COL_CNT,beta);

                // copy out
                AscendC::LocalTensor<float> outcol = outQueue.Deque<float>();
                AscendC::DataCopyPad(uGm[i * m_ + j], outcol, {len, 4, 0, m_ * sizeof(float)});

                outQueue.FreeTensor(outcol);
            }
        }
        //get Vt
        for(int32_t i=n_-2;i>=0;i--)
        {
            //the i-th householder vector updates n_-i rows,n-i-1 columns of Vt
            const int32_t len=n_-i-1;
            const int32_t padLen = len % 8 == 0 ? 0 : 8 - len % 8, ttl = len + padLen;
            auto beta=taup(i);
            if(beta==0)
            {
                continue;
            }

            // 加载当前houseVec
            row = inQueue.AllocTensor<float>();
            AscendC::DataCopyPad(row, aGm[i * n_ + i + 1], {1, len * 4, 0, 0}, {true, 0, padLen, 0.0f});
            inQueue.EnQue(row);
            row = inQueue.DeQue<float>();
            AscendC::Adds(houseVec, row, .0f, ttl);
            houseVec(0) = 1.0f;
            inQueue.FreeTensor(row);

            for (int32_t j = i; j < n_; j++)
            {
                row = inQueue.AllocTensor<float>();
                AscendC::DataCopyPad(row, vtGm[j * n_ + i+1], {1, len * 4, 0, 0}, {true, 0, padLen, 0.0f});
                inQueue.EnQue(row);

                ApplyTransformCore(ttl, beta);

                // copy out
                AscendC::LocalTensor<float> outrow = outQueue.Deque<float>();
                AscendC::DataCopyPad(vtGm[j * n_ + i+1], outrow, {1, len * 4, 0, 0});
                outQueue.FreeTensor(outrow);
            }
        }
    }
    __aicore__ inline void initUV(GM_ADDR u, GM_ADDR vt)
    {
        auto addr = reinterpret_cast<float *>(u);
        for (int32_t i = 0; i < m_; i++)
        {
            addr[i * m_ + i] = 1.0f;
        }
        addr = reinterpret_cast<float *>(vt);
        for (int32_t i = 0; i < n_; i++)
        {
            addr[i * n_ + i] = 1.0f;
        }
    }

private:
    int32_t m_, n_, k_;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    // AscendC::TBuf<AscendC::TPosition::VECCALC> colBuf, rowBuf, outBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> d_, e_, tauq_, taup_, householder_vec, scalarTmp, vecTmp;
    // AscendC::TBuf<AscendC::TPosition::VECCALC> worktmp;
    AscendC::GlobalTensor<float> aGm, uGm, vtGm;
    AscendC::LocalTensor<float> col, row, houseVec, scalartmp, vectmp, taup, tauq;
};

extern "C" __global__ __aicore__ void upper_bidiagonalization(GM_ADDR a, GM_ADDR u, GM_ADDR vt, int M, int N)
{
    Kernel_Golub_Kahan_Bidiagonalization kernel;
    kernel.Init(a, u, vt, M, N);
    kernel.Process();
}
