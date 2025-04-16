cmake_minimum_required(VERSION 3.16)
project(svd)
set(CMAKE_CXX_STANDARD 20)
set(SOC_VERSION $ENV{SOC_VERSION} CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH} CACHE PATH "ASCEND CANN package installation directory")
#below is for debug
# message("ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

#below is testing target
set(GData ../input/A_gm.bin ../output/B_golden.bin ../output/U_golden.bin ../output/Vt_golden.bin)
add_custom_command(OUTPUT ${GData} COMMAND rm -rf ${GData} && python3 ../scripts/gen_data.py DEPENDS ../args.txt ../scripts/gen_data.py ../scripts/bidiag_gen.f90)
add_custom_target(gen_data DEPENDS ${GData})

# add_library(svd_tiling STATIC svd_custom_tiling.cpp)

ascendc_library(svd_kernels STATIC svd.cpp)
# ascendc_compile_definitions(svd_kernels PRIVATE HAVE_WORKSPACE)
# ascendc_compile_definitions(svd_kernels PRIVATE HAVE_TILING)
ascendc_compile_options(svd_kernels PRIVATE "-Wno-deprecated-declarations")
ascendc_include_directories(svd_kernels PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)

ascendc_library(upper_bidiagonalization STATIC upper_bidiagonalization.cpp)
ascendc_compile_options(upper_bidiagonalization PRIVATE "-Wno-deprecated-declarations")
ascendc_include_directories(upper_bidiagonalization PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)


add_executable(svd_main main.cpp)
# target_include_directories(svd_main PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)

target_link_libraries(svd_main PRIVATE 
    svd_kernels
    upper_bidiagonalization
    tiling_api
    platform
    register
)
