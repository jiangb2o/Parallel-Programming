# 对所有生成的 massif.out.* 文件执行 ms_print
for f in massif.out.*; do
    ms_print "$f" >> valgrind.txt
done