if [ "$1" = "r" ]; then
    echo clean!
    # 清理之前的构建
    rm -rf ~/paco-svd/cmake-build-debug

    # 创建构建目录
    mkdir -p ~/paco-svd/cmake-build-debug
fi
if [ "$1" = "v" ]; then
 
    # cd scripts;
    cmake --build ~/paco-svd/cmake-build-debug --target gen_data
    exit 0
fi
# source ~/Ascend/ascend-toolkit/latest/bin/setenv.bash
# echo $ASCEND_HOME_PATH
cmake -DCMAKE_BUILD_TYPE=Debug -S ~/paco-svd -B ~/paco-svd/cmake-build-debug
cmake --build ~/paco-svd/cmake-build-debug --target svd_main -- -j 6
cd ~/paco-svd/cmake-build-debug;./svd_main
cd ~/paco-svd/scripts;python3 verify_result.py ../output ../output