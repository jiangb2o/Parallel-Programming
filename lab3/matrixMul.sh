#read -p "enter program name:" program_name
program_name="matrixMul_pthread"

./$program_name 1 128
./$program_name 1 256
./$program_name 1 512
./$program_name 1 1024
./$program_name 1 2048

./$program_name 2 128
./$program_name 2 256
./$program_name 2 512
./$program_name 2 1024
./$program_name 2 2048

./$program_name 4 128
./$program_name 4 256
./$program_name 4 512
./$program_name 4 1024
./$program_name 4 2048

./$program_name 8 128
./$program_name 8 256
./$program_name 8 512
./$program_name 8 1024
./$program_name 8 2048

./$program_name 16 128
./$program_name 16 256
./$program_name 16 512
./$program_name 16 1024
./$program_name 16 2048
