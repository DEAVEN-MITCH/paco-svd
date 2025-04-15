cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)
project(svd)

set(SOC_VERSION $ENV{SOC_VERSION} CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH} CACHE PATH "ASCEND CANN package installation directory")
message("ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

add_custom_command(OUTPUT ../input/A_gm.bin ../input/L_gm.bin ../output/golden.bin COMMAND python3 ../scripts/gen_data.py DEPENDS ../args.txt ../scripts/gen_data.py)
add_custom_target(gen_data DEPENDS ../input/A_gm.bin ../input/L_gm.bin ../output/golden.bin)

ascendc_library(svd_kernels STATIC svd.cpp)
# add_library(svd_tiling STATIC svd_custom_tiling.cpp)
add_executable(svd_main main.cpp)

# ascendc_compile_definitions(svd_kernels PRIVATE HAVE_WORKSPACE)
# ascendc_compile_definitions(svd_kernels PRIVATE HAVE_TILING)
ascendc_compile_options(svd_kernels PRIVATE "-Wno-deprecated-declarations")

ascendc_include_directories(svd_kernels PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)


# target_include_directories(svd_main PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)

target_link_libraries(svd_main PRIVATE 
    svd_kernels
    tiling_api
    platform
    register
)
