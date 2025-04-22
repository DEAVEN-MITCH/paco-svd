msprof_op_args="--output=/home/zhangjiahe/paco-svd/profing_data  "
msprof_args="--output=/home/zhangjiahe/paco-svd/profing_data --type=text --storage-limit=1024MB"
# "--kernel-name=upper_bidiagonalization" 
# "--type=text --storage-limit=1024MB"
if [ $1 = r ]; then
    cd cmake-build-release;msprof op $msprof_op_args ./svd_main
elif [ $1 = d ]; then
    cd cmake-build-debug;msprof op $msprof_op_args ./svd_main
# elif [ $1 = rall ]; then
#     cd cmake-build-release;msprof $msprof_args ./svd_main
# elif [ $1 = dall ]; then
#     cd cmake-build-debug;msprof $msprof_args ./svd_main
fi
