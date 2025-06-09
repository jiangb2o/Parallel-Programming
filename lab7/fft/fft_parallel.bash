mpicxx fft_parallel.cpp -o fft_parallel
mpirun -np 1 ./fft_parallel
mpirun -np 2 ./fft_parallel
mpirun -np 4 ./fft_parallel
mpirun --oversubscribe -np 8 ./fft_parallel
mpirun --oversubscribe -np 16 ./fft_parallel