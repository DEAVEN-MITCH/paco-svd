#include <chrono>
#include <limits>

#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils.h"

#include "acl/acl.h"

#define debug(x) std::cerr << #x << ": " << x << std::endl
int main(int argc, char **argv)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int deviceId, M, N;
    deviceId = 0;
    std::ifstream args_file("../args.txt");
    std::string m_str, n_str;
    args_file >> m_str >> n_str;
    M = std::stoi(m_str);
    N = std::stoi(n_str);
    int AicNum = ascendcPlatform->GetCoreNumAic();
    int AivNum = ascendcPlatform->GetCoreNumAiv();
    size_t UMatrixFileSize = M * M * sizeof(float);
    size_t AMatrixFileSize = M * N * sizeof(float);
    size_t VtMatrixFileSize = N * N * sizeof(float);
    // size_t userWorkspaceSize = M * N * blockDim * sizeof(float);
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    // size_t workspaceSize = M * N * sizeof(float);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *aMatrixHost;
    uint8_t *aMatrixDevice;
    // CHECK_ACL(aclrtMallocHost((void **)(&aMatrixHost), aMatrixFileSize));
    // CHECK_ACL(aclrtMalloc((void **)&aMatrixDevice, aMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // ReadFile("../input/A_gm.bin", aMatrixFileSize, aMatrixHost, aMatrixFileSize);
    // CHECK_ACL(aclrtMemcpy(aMatrixDevice, aMatrixFileSize, aMatrixHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // uint8_t *lMatrixHost;
    // uint8_t *lMatrixDevice;
    // CHECK_ACL(aclrtMallocHost((void **)(&lMatrixHost), lMatrixFileSize));
    // CHECK_ACL(aclrtMalloc((void **)&lMatrixDevice, lMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // ReadFile("../input/L_gm.bin", lMatrixFileSize, lMatrixHost, lMatrixFileSize);
    // CHECK_ACL(aclrtMemcpy(lMatrixDevice, lMatrixFileSize, lMatrixHost, lMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // uint8_t *TilingHost;
    // uint8_t *TilingDevice;
    // CHECK_ACL(aclrtMallocHost((void **)(&TilingHost), TilingFileSize));
    // CHECK_ACL(aclrtMalloc((void **)&TilingDevice, TilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // GenerateTiling(M, N, blockDim, TilingHost, lim);
    // CHECK_ACL(aclrtMemcpy(TilingDevice, TilingFileSize, TilingHost, TilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // uint8_t *xMatrixHost;
    // uint8_t *xMatrixDevice;
    // CHECK_ACL(aclrtMallocHost((void **)(&xMatrixHost), xMatrixFileSize));
    // CHECK_ACL(aclrtMalloc((void **)&xMatrixDevice, xMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // uint8_t *tMatrixHost;
    // uint8_t *tMatrixDevice;
    // CHECK_ACL(aclrtMallocHost((void **)(&tMatrixHost), aMatrixFileSize));
    // CHECK_ACL(aclrtMalloc((void **)&tMatrixDevice, aMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // ACLRT_LAUNCH_KERNEL(custom_trsm)
    // (blockDim, stream, aMatrixDevice, lMatrixDevice, xMatrixDevice, tMatrixDevice, workspaceDevice, TilingDevice);
    // CHECK_ACL(aclrtSynchronizeStream(stream));

    // std::cout << "finish" << std::endl;

    // CHECK_ACL(aclrtMemcpy(xMatrixHost, xMatrixFileSize, xMatrixDevice, xMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // // WriteFile(output_file, xMatrixHost, xMatrixFileSize);
    // CHECK_ACL(aclrtMemcpy(tMatrixHost, xMatrixFileSize, tMatrixDevice, xMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // WriteFile(output_file, xMatrixHost, xMatrixFileSize);

    // // CHECK_ACL(aclrtMemcpy(tMatrixHost, xMatrixFileSize, tMatrixDevice, xMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // std::cout << "[MATRIX X]" << std::endl;
    // PrintPartOfMatrix<float>(xMatrixHost, M, N, 8, 8);

    // CHECK_ACL(aclrtFree(aMatrixDevice));
    // CHECK_ACL(aclrtFreeHost(aMatrixHost));
    // CHECK_ACL(aclrtFree(lMatrixDevice));
    // CHECK_ACL(aclrtFreeHost(lMatrixHost));
    // CHECK_ACL(aclrtFree(xMatrixDevice));
    // CHECK_ACL(aclrtFreeHost(xMatrixHost));
    // CHECK_ACL(aclrtFree(tMatrixDevice));
    // CHECK_ACL(aclrtFreeHost(tMatrixHost));
    // CHECK_ACL(aclrtFree(TilingDevice));
    // CHECK_ACL(aclrtFreeHost(TilingHost));
    // CHECK_ACL(aclrtFree(workspaceDevice));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return 0;
}
