cmake_minimum_required(VERSION 3.16)
project(svd)
set(CMAKE_CXX_STANDARD 17)
set(SOC_VERSION $ENV{SOC_VERSION} CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH} CACHE PATH "ASCEND CANN package installation directory")
#below is for debug
# message("ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
# message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

#below is testing target
set(GData ../input/A_gm.bin ../output/B_golden.bin ../output/U_golden.bin ../output/Vt_golden.bin)
add_custom_command(OUTPUT ${GData} COMMAND rm -rf ${GData} && python3 ../scripts/gen_data.py DEPENDS ../args.txt ../scripts/gen_data.py ../scripts/bidiag_gen.f90)
add_custom_target(gen_data DEPENDS ${GData})

add_library(svd_tiling STATIC svd_tiling.cpp)
target_include_directories(svd_tiling PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw/kernel_tiling)
target_link_libraries(svd_tiling INTERFACE platform register tiling_api)

ascendc_library(svd_kernels STATIC svd.cpp)
ascendc_compile_definitions(svd_kernels PRIVATE HAVE_WORKSPACE)
ascendc_compile_definitions(svd_kernels PRIVATE HAVE_TILING)
ascendc_compile_options(svd_kernels PRIVATE "-Wno-deprecated-declarations")
# 如果CMAKE_BUILD_TYPE=Release，则不添加ASCENDC_DEBUG定义
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    ascendc_compile_definitions(svd_kernels PRIVATE -DASCENDC_DEBUG)
endif()
ascendc_include_directories(svd_kernels PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)

ascendc_library(upper_bidiagonalization STATIC upper_bidiagonalization.cpp)
# 如果CMAKE_BUILD_TYPE=Release，则不添加ASCENDC_DEBUG定义
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    ascendc_compile_definitions(upper_bidiagonalization PRIVATE -DASCENDC_DEBUG)
endif()

# ascendc_compile_definitions(upper_bidiagonalization PRIVATE -DASCENDC_DEBUG)
ascendc_compile_options(upper_bidiagonalization PRIVATE "-Wno-deprecated-declarations" -DHAVE_WORKSPACE)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    string(TIMESTAMP CURRENT_TIME "%Y%m%d_%H%M%S")
    ascendc_compile_options(upper_bidiagonalization PRIVATE -g)
    ascendc_compile_options(upper_bidiagonalization PRIVATE --cce-auto-sync-log=cce-auto-sync-log_${CURRENT_TIME}.txt)
endif()


ascendc_include_directories(upper_bidiagonalization PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)


add_executable(svd_main main.cpp)
# target_include_directories(svd_main PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)

target_link_libraries(svd_main PRIVATE 
    svd_kernels
    upper_bidiagonalization
    svd_tiling
    # platform register tiling_api
    -L${ASCEND_CANN_PACKAGE_PATH}/lib64
    -lmsprofiler
)
