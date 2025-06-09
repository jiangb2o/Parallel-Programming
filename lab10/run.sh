nvcc matMul.cu -o matMul

./matMul 128 8
./matMul 128 16
./matMul 128 32

./matMul 256 8
./matMul 256 16
./matMul 256 32

./matMul 512 8
./matMul 512 16
./matMul 512 32

./matMul 1024 8
./matMul 1024 16
./matMul 1024 32

./matMul 2048 8
./matMul 2048 16
./matMul 2048 32