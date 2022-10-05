rm *.o
opt=' -g -gencode arch=compute_75,code=sm_75 -Xcompiler -rdynamic -lineinfo -ccbin g++ --default-stream per-thread'
#opt='-g -gencode arch=compute_75,code=sm_75'
#CPP=nvcc
CPP=/usr/local/cuda/bin/nvcc
#CPP=nvcc
#opt=' -g -Xcompiler -rdynamic -lineinfo -ccbin g++ --default-stream per-thread'
exe=a.out
main=main
#main=straightSearch_test
mod=modules_cuda_clean
$CPP -c -I. $main.cu $mod.cu host.cu reads.cu $opt 
$CPP -o $exe $mod.o reads.o host.o $main.o $opt
