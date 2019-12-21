#include "ising.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define epsilon 1e-6

#define BLK_SZ 512

__device__ int calculatePos(int pos, int n, int xOffset, int yOffset)
{
    // Find coordinates
    int i = pos/n;
    int j = pos%n;
    // Perform modular arithmetic for the result
    return ((n + (i + yOffset))%n)*n + (n + (j + xOffset))%n;  
}

__global__ void swapPtr(int** ptr1, int** ptr2)
{
    int *tempPtr = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tempPtr;
}

__global__ void calculateSpin(int *current, int *next, float *w, int n)
{
    int index = blockIdx.x * BLK_SZ + threadIdx.x;
    
    if (index < n)
    {
    float result = 0.0;
    for(int i=-2; i<=2; i++)
        for(int j=-2; j<=2; j++)
        {
            result += w[ (i+2)*5 + (j+2) ]*current[ calculatePos(blockIdx.x, n, j, i) ];
        }
    if( result < epsilon && result > -epsilon)
        next[index] = current[index];
    else if (result < 0)
        next[index] = -1;
    else
        next[index] = 1;
    }
}

void ising(int *G, float *w, int k, int n)
{
    int *dev_G, *dev_G2;
    float *dev_w;

    cudaError err;
    
    // Memory Allocation and Memory Copy on Device
    err = cudaMalloc((void**)&dev_G, (size_t)n*n*sizeof(int));
    printf("Allocate 1: %s\n",cudaGetErrorString(err));
    err = cudaMalloc((void**)&dev_G2, (size_t)n*n*sizeof(int));
    printf("Allocate 2: %s\n",cudaGetErrorString(err));
    err = cudaMalloc((void**)&dev_w, 25*sizeof(float));
    printf("Allocate: %s\n",cudaGetErrorString(err));
    err = cudaMemcpy(dev_G, G, (size_t)n*n*sizeof(int), cudaMemcpyHostToDevice);
    printf("Send 1: %s\n",cudaGetErrorString(err));
    err = cudaMemcpy(dev_w, w, 25*sizeof(float), cudaMemcpyHostToDevice);
    printf("Send 2: %s\n",cudaGetErrorString(err));

    // Make the Calculations

    printf("Blocks: %d, Threads: %d\n", (n*n)/BLK_SZ + 1, BLK_SZ);
    calculateSpin<<<(n*n)/BLK_SZ + 1, BLK_SZ>>>(dev_G, dev_G2, dev_w, n);
    err = cudaGetLastError();
    printf("Launch: %s\n",cudaGetErrorString(err));

    err = cudaThreadSynchronize();
    printf("Sync: %s\n",cudaGetErrorString(err));
    //swapPtr<<<1,1>>>(&dev_G, &dev_G2);
    //cudaDeviceSynchronize();
    // Copy Results back to Host
    err = cudaMemcpy(G, dev_G2, (size_t)n*n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("Send 3: %s\n",cudaGetErrorString(err));
    // Memory Cleanup
    cudaFree(dev_G);
    cudaFree(dev_G2);
}
