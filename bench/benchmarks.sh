#!/usr/bin/fish

make benchmark_sequential benchmark_cudaV1 benchmark_cudaV2 benchmark_cudaV3

set n 50 100 300 500 1000 1500 2000 3000 5000 
set k 5 10 20 50
set inputs $n" "$k

echo $inputs | xargs -n 2 ./benchmark_sequential
echo $inputs | xargs -n 2 ./benchmark_cudaV1
echo $inputs | xargs -n 2 ./benchmark_cudaV2 
echo $inputs | xargs -n 2 ./benchmark_cudaV3 

make clean