#include "kernel_operator.h"
constexpr int BUFFER_NUM = 2;
constexpr int maxL1FloatSize = 1 << 19 - 2;
constexpr int BlockSize = 32;
constexpr int CONCURRENT_COL_CNT = BlockSize / sizeof(float);
constexpr int SizePerOperation = 256;
constexpr int BlockNumPerOperation = SizePerOperation / BlockSize;
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
        initUV(u, vt, M, N);

        // 设置全局内存缓冲区
        aGm.SetGlobalBuffer((__gm__ float *)a, M * N);
        uGm.SetGlobalBuffer((__gm__ float *)u, M * M);
        vtGm.SetGlobalBuffer((__gm__ float *)vt, N * N);

        // 初始化管道缓冲区
        pipe.InitBuffer(colQueue, BUFFER_NUM, M * BlockSize + SizePerOperation);
        pipe.InitBuffer(rowQueue, BUFFER_NUM, N * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, M * BlockSize + SizePerOperation);
        pipe.InitBuffer(d_, N * sizeof(float));
        pipe.InitBuffer(e_, (N - 1) * sizeof(float));
        pipe.InitBuffer(tauq_, N * sizeof(float));
        pipe.InitBuffer(taup_, (N - 1) * sizeof(float));
        pipe.InitBuffer(householder_vec, (M - 1) * BlockSize + SizePerOperation); // an additional SizePerOperation in case tail masked operation exceeds the address boundary
        // reduce sum need at most M
        pipe.InitBuffer(worktmp, (M - 1) * BlockSize + SizePerOperation);
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

        // 构造双对角矩阵
        ConstructBidiagonalMatrix();
        GetUVt();
    }

private:
    __aicore__ inline void ComputeColumnHouseholder(int32_t i)
    {
        assert(i < n_ && "in ComputeColumnHouseholder, i should be less than n_");
        const int32_t len = _m - i;
        if (len <= 1)
        {
            // a scalar vector, no need to calc,beta=0;
            tauq_(i) = 0;
            return;
        }
        AscendC::LocalTensor<float> col = colQueue.AllocTensor<float>();
        const auto floor_i = i / CONCURRENT_COL_CNT, offset_i = i % CONCURRENT_COL_CNT;
        uint64_t mask[1] = {MASK_PATTERN << offset_i};
        const uint8_t repeatTimes = (len - 1) / BlockNumPerOperation, tailSize = (len - 1) % BlockNumPerOperation;
        const uint64_t tailOffset = repeatTimes * SizePerOperation + CONCURRENT_COL_CNT;
        // if tailSize > 0,only tailSize个 bit is 1
        uint64_t tailMask[1] = {MASK_PATTERN << (CONCURRENT_COL_CNT * (BlockNumPerOperation - tailSize))};
        // 加载当前列,floor_i 列的block，加载M-i块
        AscendC::DataCopy(col, aGm[floor_i * n_], {len, 1, n_ / CONCURRENT_COL_CNT, 0});
        colQueue.EnQue(col);

        // Vector Compute
        col = colQueue.DeQue<float>();
        AscendC::LocalTensor<float> outcol = householder_vec.Get<float>();
        AscendC::LocalTensor<float> tmp = worktmp.Get<float>();
        // pad the tail operation blocks with 0
        AscendC::Duplicate(col[len * CONCURRENT_COL_CNT], .0f, BlockNumPerOperation * CONCURRENT_COL_CNT);
        // copy col to outcol
        AscendC::Adds(outcol, col[CONCURRENT_COL_CNT], .0f, (len-1)*CONCURRENT_COL_CNT);

        // get the first element of x 
        auto x1 = col(offset_i);
        // calculate the 2-norm of x except the first element
        // round up repeatTimes to deal the tail together
        AscendC::Mul(tmp, col[CONCURRENT_COL_CNT], col[CONCURRENT_COL_CNT], mask, (len - 1 + BlockNumPerOperation - 1) / BlockNumPerOperation, {1, 1, 1, BlockNumPerOperation, BlockNumPerOperation, BlockNumPerOperation});
        // use col as tmp worklocal space,col[0]=x(2:)Tx(2:)
        AscendC::ReduceSum(col, tmp, col[CONCURRENT_COL_CNT], mask, (len - 1 + BlockNumPerOperation - 1) / BlockNumPerOperation, BlockNumPerOperation);
        // update tauq_[i] as beta,
        auto sigma=col(0);
        colQueue.FreeTensor(col);
        if(sigma==0){
            if(x1>=0){
                tauq_(i)=0;
            }else{
                tauq_(i)=-2;
            }
        }else{
            auto miu=std::sqrt(x1*x1+sigma);
            float v1;
            if(x1<=0){//update v1,aka,outcol(offset_i)
                v1=x1-miu;
            }else{
                v1=-sigma/(x1+miu);
            }
            float v1sq=v1*v1;
            tauq_(i)=2*v1sq/(sigma+v1sq);
            AscendC::Muls(outcol,outcol,1.0f/v1sq,mask,{1,1,BlockNumPerOperation,BlockNumPerOperation);
        }

        //output len-1 essential householder vector
        // the outcol is the essential householder vector

    }
    __aicore__ inline void ComputeRowHouseholder(int32_t i)
    {
        AscendC::LocalTensor<float> row = rowQueue.AllocTensor<float>();

        // 加载当前行
        LoadRow(row, i, i + 1);

        // 计算Householder向量
        float norm = ComputeNorm(row);
        e_[i] = (row[0] >= 0) ? -norm : norm;

        // 计算tau
        float alpha = e_[i];
        taup_[i] = (alpha - row[0]) / alpha;

        // 更新行向量
        row[0] = alpha;

        // 存储更新后的行
        StoreRow(row, i, i + 1);

        rowQueue.FreeTensor(row);
    }
    __aicore__ inline void ApplyColumnTransform(int32_t i)
    {
        // 应用列变换到右侧子矩阵
        for (int32_t j = i + 1; j < n_; j++)
        {
            UpdateColumn(i, j);
        }
    }
    __aicore__ inline void ApplyRowTransform(int32_t i)
    {
        // 应用行变换到下方子矩阵
        for (int32_t j = i + 1; j < m_; j++)
        {
            UpdateRow(i, j);
        }
    }
    __aicore__ inline void UpdateColumn(int32_t i, int32_t j)
    {
        AscendC::LocalTensor<float> col = colQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> result = outQueue.AllocTensor<float>();

        // 加载列
        LoadColumn(col, i, j);

        // 应用Householder变换
        ApplyHouseholderTransform(col, result, tauq_[i]);

        // 存储结果
        StoreColumn(result, i, j);

        colQueue.FreeTensor(col);
        outQueue.FreeTensor(result);
    }
    __aicore__ inline void UpdateRow(int32_t i, int32_t j)
    {
        AscendC::LocalTensor<float> row = rowQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> result = outQueue.AllocTensor<float>();

        // 加载行
        LoadRow(row, i, j);

        // 应用Householder变换
        ApplyHouseholderTransform(row, result, taup_[i]);

        // 存储结果
        StoreRow(result, i, j);

        rowQueue.FreeTensor(row);
        outQueue.FreeTensor(result);
    }
    __aicore__ inline void ConstructBidiagonalMatrix()
    {
        // 构造上双对角矩阵
        for (int32_t i = 0; i < k_; i++)
        {
            // 设置对角元素
            SetMatrixElement(i, i, d_[i]);

            // 设置超对角元素
            if (i < k_ - 1)
            {
                SetMatrixElement(i, i + 1, e_[i]);
            }
        }
    }
    // 辅助函数
    __aicore__ inline float ComputeNorm(const AscendC::LocalTensor<float> &vec)
    {
        float sum = 0;
        for (int32_t i = 0; i < vec.size(); i++)
        {
            sum += vec[i] * vec[i];
        }
        return AscendC::Sqrt(sum);
    }
    __aicore__ inline void ApplyHouseholderTransform(
        const AscendC::LocalTensor<float> &input,
        AscendC::LocalTensor<float> &output,
        float tau)
    {
        // H = I - tau * v * v^T
        // y = H * x = x - tau * v * (v^T * x)
        float dot = 0;
        for (int32_t i = 0; i < input.size(); i++)
        {
            dot += input[i] * input[i];
        }

        for (int32_t i = 0; i < input.size(); i++)
        {
            output[i] = input[i] - tau * input[i] * dot;
        }
    }

    __aicore__ inline void StoreColumn(const AscendC::LocalTensor<float> &col, int32_t i, int32_t j)
    {
        // 将列向量存储回全局内存
        for (int32_t k = i; k < m_; k++)
        {
            aGm[k * n_ + j] = col[k - i];
        }
    }

    __aicore__ inline void LoadRow(AscendC::LocalTensor<float> &row, int32_t i, int32_t j)
    {
        // 从全局内存加载行向量
        for (int32_t k = j; k < n_; k++)
        {
            row[k - j] = aGm[i * n_ + k];
        }
    }

    __aicore__ inline void StoreRow(const AscendC::LocalTensor<float> &row, int32_t i, int32_t j)
    {
        // 将行向量存储回全局内存
        for (int32_t k = j; k < n_; k++)
        {
            aGm[i * n_ + k] = row[k - j];
        }
    }

    __aicore__ inline void SetMatrixElement(int32_t i, int32_t j, float value)
    {
        // 设置矩阵元素
        aGm[i * n_ + j] = value;
    }
     __aicore__ inline void GetUVt())
     {
     }
     __aicore__ inline void UpdateMatrixU(int32_t i)
     {
         // 更新U矩阵的第i列
         for (int32_t k = 0; k < m_; k++)
         {
             for (int32_t j = 0; j < m_; j++)
             {
                 if (k == j)
                 {
                     uGm[k * m_ + j] = 1.0;
                 }
                 else
                 {
                     uGm[k * m_ + j] = 0.0;
                 }
             }
         }
     }

     __aicore__ inline void UpdateMatrixVT(int32_t i)
     {
         // 更新VT矩阵的第i行
         for (int32_t k = 0; k < n_; k++)
         {
             for (int32_t j = 0; j < n_; j++)
             {
                 if (k == j)
                 {
                     vtGm[k * n_ + j] = 1.0;
                 }
                 else
                 {
                     vtGm[k * n_ + j] = 0.0;
                 }
             }
         }
     }
     __aicore__ inline void initUV(GM_ADDR u, GM_ADDR vt, int M, int N)
     {
         auto addr = reinterpret_cast<float *>(u);
         for (int32_t i = 0; i < M; i++)
         {
             addr[i * m_ + i] = 1.0f;
         }
         addr = reinterpret_cast<float *>(vt);
         for (int32_t i = 0; i < N; i++)
         {
             addr[i * n_ + i] = 1.0f;
         }
     }
} private : int32_t m_, n_, k_;
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> colQueue, rowQueue;
AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
AscendC::TBuf<AscendC::TPosition::VECCALC> d_, e_, tauq_, taup_, householder_vec, worktmp;
AscendC::GlobalTensor<float> aGm, uGm, vtGm;
}
;

extern "C" __global__ __aicore__ void upper_bidiagonalization(GM_ADDR a, GM_ADDR u, GM_ADDR vt, int M, int N)
{
    Kernel_Golub_Kahan_Bidiagonalization kernel;
    kernel.Init(a, u, vt, M, N);
    kernel.Process();
}
