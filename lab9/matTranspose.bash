nvcc matTranspose.cu -o matTranspose

./matTranspose 512 4
./matTranspose 1024 4
./matTranspose 2048 4

./matTranspose 512 8
./matTranspose 1024 8
./matTranspose 2048 8

./matTranspose 512 16
./matTranspose 1024 16
./matTranspose 2048 16

./matTranspose 512 32
./matTranspose 1024 32
./matTranspose 2048 32


./matTranspose 512 4 1
./matTranspose 1024 4 1
./matTranspose 2048 4 1

./matTranspose 512 8 1
./matTranspose 1024 8 1
./matTranspose 2048 8 1

./matTranspose 512 16 1
./matTranspose 1024 16 1
./matTranspose 2048 16 1

./matTranspose 512 32 1
./matTranspose 1024 32 1
./matTranspose 2048 32 1