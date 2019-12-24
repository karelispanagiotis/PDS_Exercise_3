#include "ising.h"
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define epsilon 1e-6f

#define BLK_SZ 512

__device__ __forceinline__ 
int mod(int x, int y)
{
    return (y + x%y)%y;
}

__device__  __forceinline__ 
int calcLatticePos(int pos, int n, int xOffset, int yOffset)
{
    /* Finds the index in the lattice, according to
     *  pos: Current Position (in which we calculate spin)
     *  xOffset: Offset in the x-axis
     *  yOffset: Offset in the y-axis
     * -----------------------------------------------------
     *  pos/n : is the row that corresponds to pos
     *  pos%n : is the column that corresponds to pos
     */

    // Perform Modular Arithmetic
    return mod(pos/n + yOffset, n)*n + mod(pos%n + xOffset, n);  
}

void swapPtr(int** ptr1, int** ptr2)
{
    int *tempPtr = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tempPtr;
}

__global__ void calculateSpin(int *current, int *next, float *w, int n)
{
    size_t index = (size_t)blockIdx.x * BLK_SZ + threadIdx.x;
    
    if(index < (size_t)n*n)
    {
        float result = 0.0f;
        int i,j;
        for(i=-2; i<=2; i++)
            for(j=-2; j<=2; j++)
                result += w[ (i+2)*5 + (j+2) ] * current[ calcLatticePos(index, n, j, i) ];
        
        if(fabsf(result) < epsilon )
            next[index] = current[index];
        else if(result < 0)
            next[index] = -1;
        else
            next[index] = 1;
    }
    
    
}

void ising(int *G, float *w, int k, int n)
{
    int *dev_G, *dev_G2;
    float *dev_w;
    size_t latticeSize = (size_t)n*n*sizeof(int);

    // Data Transfer and Memory Alloc on Device
    cudaMalloc(&dev_G, latticeSize);
    cudaMalloc(&dev_G2, latticeSize);
    cudaMalloc(&dev_w, 25*sizeof(float));

    cudaMemcpy(dev_G, G, latticeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_w, w, 25*sizeof(float), cudaMemcpyHostToDevice);

    // Kernel Launch - Calculations
    for(int iter=0; iter<k; iter++)
    {
        calculateSpin<<<((size_t)n*n + BLK_SZ - 1)/BLK_SZ, BLK_SZ>>>(dev_G, dev_G2, dev_w, n);
        cudaDeviceSynchronize();
        swapPtr(&dev_G, &dev_G2);
    }

    // Send Result back to CPU
    cudaMemcpy(G, dev_G, latticeSize, cudaMemcpyDeviceToHost);

    // Clear Resources
    cudaFree(dev_G);
    cudaFree(dev_G2);
}
