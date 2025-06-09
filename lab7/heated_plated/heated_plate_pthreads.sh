# compile with libparallel.so
g++ heated_plate_pthreads.cpp -L. -lparallel -o heated_plate_pthreads
# 将当前目录加入库搜索路径
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH