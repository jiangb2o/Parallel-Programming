valgrind --tool=massif --stacks=yes ./heated_plate_pthreads 1 128
valgrind --tool=massif --stacks=yes ./heated_plate_pthreads 2 128
valgrind --tool=massif --stacks=yes ./heated_plate_pthreads 4 128
valgrind --tool=massif --stacks=yes ./heated_plate_pthreads 8 128
valgrind --tool=massif --stacks=yes ./heated_plate_pthreads 16 128

valgrind --tool=massif --stacks=yes ./heated_plate_pthreads 1 256
valgrind --tool=massif --stacks=yes ./heated_plate_pthreads 2 256
valgrind --tool=massif --stacks=yes ./heated_plate_pthreads 4 256
valgrind --tool=massif --stacks=yes ./heated_plate_pthreads 8 256
valgrind --tool=massif --stacks=yes ./heated_plate_pthreads 16 256

valgrind --tool=massif --stacks=yes  ./heated_plate_pthreads 1 512
valgrind --tool=massif --stacks=yes  ./heated_plate_pthreads 2 512
valgrind --tool=massif --stacks=yes  ./heated_plate_pthreads 4 512
valgrind --tool=massif --stacks=yes  ./heated_plate_pthreads 8 512
valgrind --tool=massif --stacks=yes  ./heated_plate_pthreads 16 512



