# compile with libparallel.so
g++ use_parallel_for.cpp -L. -lparallel -o use_parallel_for
g++ matMul.cpp -L. -lparallel -o matMul
# 将当前目录加入库搜索路径
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH