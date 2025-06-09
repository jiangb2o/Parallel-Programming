nvcc main.cu SlideConvolution.cu Im2colConvolution.cu cuDNN.cu -o Convolution -lcudnn

rm -rf ./result
mkdir result

# default parameter:
# Input channel: 3
# Kernel size: 3 x 3
# Kernel channel: 3
# Stride: 1,2,3

# parameter1: Thread block size  
# parameter2: Input matrix size(32-512)  

./Convolution 32 32  
./Convolution 32 64  
./Convolution 32 128 
./Convolution 32 256 
./Convolution 32 512 


