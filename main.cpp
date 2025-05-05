#include <chrono>
#include <limits>

#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils.h"

#include "acl/acl.h"
#include "aclrtlaunch_upper_bidiagonalization.h"
#include "aclrtlaunch_svd_DC.h"
#define debug(x) std::cerr << #x << ": " << x << std::endl
extern int getTilingSize(int N, int svdBlockNum);
extern void GenerateTiling(int N, int blockDim, uint8_t *TilingHost);
int main(int argc, char **argv)
{
    unsigned int deviceId, M, N;
    deviceId = 1;
    unsigned int bidiagonalizationBlockNum = 40, svdBlockNum = 1;
    std::ifstream args_file("../args.txt");
    std::string m_str, n_str;
    args_file >> m_str >> n_str;
    M = std::stoi(m_str);
    N = std::stoi(n_str);
    if (M < N)
    {
        std::cerr << "M must be greater than N" << std::endl;
        return 0;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int AicNum = ascendcPlatform->GetCoreNumAic();
    int AivNum = ascendcPlatform->GetCoreNumAiv();
    size_t UMatrixFileSize = M * M * sizeof(float);
    size_t AMatrixFileSize = M * N * sizeof(float);
    size_t VtMatrixFileSize = N * N * sizeof(float);
    size_t DArrayFileSize = N * sizeof(float);
    size_t EArrayFileSize = (N - 1) * sizeof(float);
    size_t TauqArrayFileSize = N * sizeof(float);
    size_t TaupArrayFileSize = (N - 1) * sizeof(float);
    size_t QtMatrixFileSize = 2 * N * N * sizeof(float);
    size_t WtMatrixFileSize = 2 * N * N * sizeof(float);
    size_t TilingFileSize = getTilingSize(N, svdBlockNum);
    size_t idxFileSize = 16 * N * sizeof(uint32_t);

    // size_t userWorkspaceSize = M * N * blockDim * sizeof(float);
    size_t userWorkspaceSize = bidiagonalizationBlockNum * 32; // 软同步需要的GM工作空间
    size_t systemWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    // size_t workspaceSize = M * N * sizeof(float);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(workspaceDevice, userWorkspaceSize, 0, userWorkspaceSize)); // 初始化GM工作空间为0

    uint8_t *AMatrixHost;
    uint8_t *AMatrixDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&AMatrixHost), AMatrixFileSize));
    CHECK_ACL(aclrtMalloc((void **)&AMatrixDevice, AMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("../input/A_gm.bin", AMatrixFileSize, AMatrixHost, AMatrixFileSize);
    CHECK_ACL(aclrtMemcpy(AMatrixDevice, AMatrixFileSize, AMatrixHost, AMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // std::cout << "[MATRIX A]" << std::endl;
    // PrintPartOfMatrix<float>(AMatrixHost, M, N, 8, 8);

    uint8_t *TilingHost;
    uint8_t *TilingDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&TilingHost), TilingFileSize));
    CHECK_ACL(aclrtMalloc((void **)&TilingDevice, TilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    GenerateTiling(N, svdBlockNum, TilingHost);
    CHECK_ACL(aclrtMemcpy(TilingDevice, TilingFileSize, TilingHost, TilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *UMatrixHost;
    uint8_t *UMatrixDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&UMatrixHost), UMatrixFileSize));
    CHECK_ACL(aclrtMalloc((void **)&UMatrixDevice, UMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(UMatrixDevice, UMatrixFileSize, 0, UMatrixFileSize));

    uint8_t *VtMatrixHost;
    uint8_t *VtMatrixDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&VtMatrixHost), VtMatrixFileSize));
    CHECK_ACL(aclrtMalloc((void **)&VtMatrixDevice, VtMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(VtMatrixDevice, VtMatrixFileSize, 0, VtMatrixFileSize));

    uint8_t *DArrayHost;
    uint8_t *DArrayDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&DArrayHost), DArrayFileSize));
    CHECK_ACL(aclrtMalloc((void **)&DArrayDevice, DArrayFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // uint8_t *EArrayHost;
    uint8_t *EArrayDevice;
    // CHECK_ACL(aclrtMallocHost((void **)(&EArrayHost), EArrayFileSize));
    CHECK_ACL(aclrtMalloc((void **)&EArrayDevice, EArrayFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // uint8_t *TauqArrayHost;
    uint8_t *TauqArrayDevice;
    // CHECK_ACL(aclrtMallocHost((void **)(&TauqArrayHost), TauqArrayFileSize));
    CHECK_ACL(aclrtMalloc((void **)&TauqArrayDevice, TauqArrayFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // uint8_t *TaupArrayHost;
    uint8_t *TaupArrayDevice;
    // CHECK_ACL(aclrtMallocHost((void **)(&TaupArrayHost), TaupArrayFileSize));
    CHECK_ACL(aclrtMalloc((void **)&TaupArrayDevice, TaupArrayFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // uint8_t *QtMatrixHost;
    uint8_t *QtMatrixDevice;
    // CHECK_ACL(aclrtMallocHost((void **)(&QtMatrixHost), QtMatrixFileSize));
    CHECK_ACL(aclrtMalloc((void **)&QtMatrixDevice, QtMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(QtMatrixDevice, QtMatrixFileSize, 0, QtMatrixFileSize));

    // uint8_t *WtMatrixHost;
    uint8_t *WtMatrixDevice;
    // CHECK_ACL(aclrtMallocHost((void **)(&WtMatrixHost), WtMatrixFileSize));
    CHECK_ACL(aclrtMalloc((void **)&WtMatrixDevice, WtMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(WtMatrixDevice, WtMatrixFileSize, 0, WtMatrixFileSize));

    uint8_t *idxDevice;
    CHECK_ACL(aclrtMalloc((void **)&idxDevice, idxFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(idxDevice, idxFileSize, 0, idxFileSize));

    ACLRT_LAUNCH_KERNEL(upper_bidiagonalization)
    (bidiagonalizationBlockNum, stream, M, N, AMatrixDevice, UMatrixDevice, VtMatrixDevice, DArrayDevice, EArrayDevice, TauqArrayDevice, TaupArrayDevice, workspaceDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::cout << "finish bidiagonalization" << std::endl;

    CHECK_ACL(aclrtMemcpy(UMatrixHost, UMatrixFileSize, UMatrixDevice, UMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // CHECK_ACL(aclrtMemcpy(AMatrixHost, AMatrixFileSize, AMatrixDevice, AMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(VtMatrixHost, VtMatrixFileSize, VtMatrixDevice, VtMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // CHECK_ACL(aclrtMemcpy(DArrayHost, DArrayFileSize, DArrayDevice, DArrayFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // CHECK_ACL(aclrtMemcpy(EArrayHost, EArrayFileSize, EArrayDevice, EArrayFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // CHECK_ACL(aclrtMemcpy(TauqArrayHost, TauqArrayFileSize, TauqArrayDevice, TauqArrayFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // CHECK_ACL(aclrtMemcpy(TaupArrayHost, TaupArrayFileSize, TaupArrayDevice, TaupArrayFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // {
    //     // construct B matrix from D and E
    //     CHECK_ACL(aclrtMemset(AMatrixHost, AMatrixFileSize, 0, AMatrixFileSize));
    //     for (int32_t i = 0; i < N; i++)
    //     {
    //         reinterpret_cast<float *>(AMatrixHost)[i * N + i] = reinterpret_cast<float *>(DArrayHost)[i];
    //         if (i < N - 1)
    //             reinterpret_cast<float *>(AMatrixHost)[i * N + i + 1] = reinterpret_cast<float *>(EArrayHost)[i];
    //     }
    // }
    // WriteFile("../output/U.bin", UMatrixHost, UMatrixFileSize);
    // WriteFile("../output/B.bin", AMatrixHost, AMatrixFileSize);
    // WriteFile("../output/Vt.bin", VtMatrixHost, VtMatrixFileSize);

    // std::cout << "[MATRIX B]" << std::endl;
    // PrintPartOfMatrix<float>(AMatrixHost, M, N, 8, 8);
    // PrintPartOfMatrix<float>(UMatrixHost, M, M, 2, 2);
    // PrintPartOfMatrix<float>(VtMatrixHost, N, N, 2, 2);
    ACLRT_LAUNCH_KERNEL(svd_DC)
    (svdBlockNum, stream, M, N, AMatrixDevice, UMatrixDevice, VtMatrixDevice, DArrayDevice, EArrayDevice, QtMatrixDevice, WtMatrixDevice, idxDevice, workspaceDevice, TilingDevice);

    CHECK_ACL(aclrtSynchronizeStream(stream));
    std::cout << "finish" << std::endl;

    CHECK_ACL(aclrtMemcpy(UMatrixHost, UMatrixFileSize, UMatrixDevice, UMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(VtMatrixHost, VtMatrixFileSize, VtMatrixDevice, VtMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(DArrayHost, DArrayFileSize, DArrayDevice, DArrayFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // CHECK_ACL(aclrtMemcpy(AMatrixHost, AMatrixFileSize, AMatrixDevice, AMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // PrintPartOfMatrix<float>(AMatrixHost, M, N, 2, 2);

    {
        // 直接将D数组作为奇异值向量输出
        WriteFile("../output/sigma.bin", DArrayHost, DArrayFileSize);
    }
    WriteFile("../output/U.bin", UMatrixHost, UMatrixFileSize);
    WriteFile("../output/Vt.bin", VtMatrixHost, VtMatrixFileSize);

    CHECK_ACL(aclrtFree(AMatrixDevice));
    CHECK_ACL(aclrtFreeHost(AMatrixHost));
    CHECK_ACL(aclrtFree(UMatrixDevice));
    CHECK_ACL(aclrtFreeHost(UMatrixHost));
    CHECK_ACL(aclrtFree(VtMatrixDevice));
    CHECK_ACL(aclrtFreeHost(VtMatrixHost));
    CHECK_ACL(aclrtFree(DArrayDevice));
    CHECK_ACL(aclrtFreeHost(DArrayHost));
    CHECK_ACL(aclrtFree(EArrayDevice));
    // CHECK_ACL(aclrtFreeHost(EArrayHost));
    CHECK_ACL(aclrtFree(TauqArrayDevice));
    // CHECK_ACL(aclrtFreeHost(TauqArrayHost));
    CHECK_ACL(aclrtFree(TaupArrayDevice));
    // CHECK_ACL(aclrtFreeHost(TaupArrayHost));
    CHECK_ACL(aclrtFree(QtMatrixDevice));
    // CHECK_ACL(aclrtFreeHost(QtMatrixHost));
    CHECK_ACL(aclrtFree(WtMatrixDevice));
    // CHECK_ACL(aclrtFreeHost(WtMatrixHost));
    // CHECK_ACL(aclrtFree(TilingDevice));
    // CHECK_ACL(aclrtFreeHost(TilingHost));
    CHECK_ACL(aclrtFree(workspaceDevice));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return 0;
}
