#include "ising.h"
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define BLK_DIM_SZ 32

#define WGHT_SZ 25
#define WGHT_DIM_SZ 5
#define MAX_OFFSET 2

#define epsilon 1e-6f

__device__ __forceinline__ 
int mod(int x, int y)
{
    return (y + x%y)%y;
}

__device__  __forceinline__ 
int calcLatticePos(int i, int j, int n)
{
    // Perform Modular Arithmetic
    return mod(i, n)*n + mod(j, n);   
}

void swapPtr(int** ptr1, int** ptr2)
{
    int *tempPtr = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tempPtr;
}

__global__ void calculateSpin(int *current, int *next, float *w, int n)
{
    int gindex_i = blockIdx.y*BLK_DIM_SZ + threadIdx.y; 
    int gindex_j = blockIdx.x*BLK_DIM_SZ + threadIdx.x;
    int lindex_i = threadIdx.y + MAX_OFFSET;
    int lindex_j = threadIdx.x + MAX_OFFSET; 

    __shared__ int spinsBlock[ BLK_DIM_SZ + 2*MAX_OFFSET ] [ BLK_DIM_SZ + 2*MAX_OFFSET ];
    __shared__ float weights[ WGHT_DIM_SZ ] [ WGHT_DIM_SZ ];

    // Fetch Data from Device/Global Memory into Shared Memory

    // Populate Inner Array (no Borders)
    spinsBlock[lindex_i][lindex_j] = current[ calcLatticePos(gindex_i, gindex_j, n) ];

    // Populate Left and Right Borders (except Corners)
    if(threadIdx.x < MAX_OFFSET)
    {
        spinsBlock[lindex_i][lindex_j - MAX_OFFSET] = current[ calcLatticePos(gindex_i, gindex_j - MAX_OFFSET, n) ];
        spinsBlock[lindex_i][lindex_j + BLK_DIM_SZ] = current[ calcLatticePos(gindex_i, gindex_j + BLK_DIM_SZ, n) ];
    }

    // Populate Upper and Lower Borders (except Corners)
    if(threadIdx.y < MAX_OFFSET)
    {
        spinsBlock[lindex_i - MAX_OFFSET][lindex_j] = current[ calcLatticePos(gindex_i - MAX_OFFSET, gindex_j, n) ]; 
        spinsBlock[lindex_i + BLK_DIM_SZ][lindex_j] = current[ calcLatticePos(gindex_i + BLK_DIM_SZ, gindex_j, n) ];
    }

    // Populate the Four Corners
    if( (threadIdx.x > BLK_DIM_SZ - MAX_OFFSET - 1) && (threadIdx.y > BLK_DIM_SZ - MAX_OFFSET - 1) )
    {
        spinsBlock[lindex_i + MAX_OFFSET][lindex_j + MAX_OFFSET]    //bottom-right corner
            = current[ calcLatticePos(gindex_i + MAX_OFFSET, gindex_j + MAX_OFFSET, n) ];
        
        spinsBlock[lindex_i + MAX_OFFSET][lindex_j - BLK_DIM_SZ]    //bottom-left corner
            = current[ calcLatticePos(gindex_i + MAX_OFFSET, gindex_j - BLK_DIM_SZ, n) ];

        spinsBlock[lindex_i - BLK_DIM_SZ][lindex_j - BLK_DIM_SZ]    //upper-left corner
            = current[ calcLatticePos(gindex_i - BLK_DIM_SZ, gindex_j - BLK_DIM_SZ, n) ];

        spinsBlock[lindex_i - BLK_DIM_SZ][lindex_j + MAX_OFFSET]    //upper-right corner
            = current[ calcLatticePos(gindex_i - BLK_DIM_SZ, gindex_j + MAX_OFFSET, n) ];
    }

    // Fetch Weights Data
    if( threadIdx.x<WGHT_DIM_SZ && threadIdx.y<WGHT_DIM_SZ )
        weights[threadIdx.y][threadIdx.x] = w[threadIdx.y*WGHT_DIM_SZ + threadIdx.x];

    __syncthreads();

    if(gindex_i<n && gindex_j<n)
    {
        float result = 0.0f;
        int i,j;
        for(i=-MAX_OFFSET; i<=MAX_OFFSET; i++)
            for(j=-MAX_OFFSET; j<=MAX_OFFSET; j++)
                result += weights[i + MAX_OFFSET][j + MAX_OFFSET] * spinsBlock[lindex_i + i][lindex_j + j];
        
        int gindex = gindex_i*n + gindex_j;
        if(fabsf(result) < epsilon )
            next[gindex] = spinsBlock[lindex_i][lindex_j];
        else if(result < 0)
            next[gindex] = -1;
        else
            next[gindex] = 1;

    }
}

/* For this version of CUDA Ising:
 * Each CUDA block of threads calculates
 * one BLK_DIM_SZ x BLK_DIM_SZ block in the spin lattice.
 */
void ising(int *G, float *w, int k, int n)
{
    int *dev_G, *dev_G2;
    float *dev_w;
    size_t latticeSize = (size_t)n*n*sizeof(int);

    // Data Transfer and Memory Alloc on Device
    cudaMalloc(&dev_G, latticeSize);
    cudaMalloc(&dev_G2, latticeSize);
    cudaMalloc(&dev_w, WGHT_SZ*sizeof(float));

    cudaMemcpy(dev_G, G, latticeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_w, w, WGHT_SZ*sizeof(float), cudaMemcpyHostToDevice);

    // Kernel Launch - Calculations
    int gridDimSz = (n + BLK_DIM_SZ - 1)/BLK_DIM_SZ;
    dim3 grid2D(gridDimSz, gridDimSz);
    dim3 block2D(BLK_DIM_SZ, BLK_DIM_SZ);
    for(int iter=0; iter<k; iter++)
    {
        calculateSpin<<<grid2D, block2D>>>(dev_G, dev_G2, dev_w, n);
        cudaDeviceSynchronize();
        swapPtr(&dev_G, &dev_G2);
    }

    // Send Result back to CPU
    cudaMemcpy(G, dev_G, latticeSize, cudaMemcpyDeviceToHost);

    // Clear Resources
    cudaFree(dev_G);
    cudaFree(dev_G2);
}
