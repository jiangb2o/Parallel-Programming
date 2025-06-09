# 设置测试参数
NUM_OPERATIONS=30000000  # 进行的浮点运算次数
START_TIME=$(date +%s.%N)  # 记录开始时间

# 执行浮点运算
echo "scale=10; for (i=0; i<$NUM_OPERATIONS; ++i) {sqrt(i^2) * atan(1)}" | bc -l > /dev/null

# 计算总时间
END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# 计算FLOPS并转换为GFLOPS
FLOPS=$(echo "scale=10; $NUM_OPERATIONS / $ELAPSED_TIME" | bc)
GFLOPS=$(echo "scale=10; $FLOPS / 1000000000" | bc)

echo "总共执行了 $NUM_OPERATIONS 次浮点运算"
echo "总耗时: $ELAPSED_TIME 秒"
echo "平均每秒浮点运算次数: $FLOPS FLOPS"
echo "峰值性能: $GFLOPS GFLOPS"
