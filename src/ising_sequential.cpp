#include "ising.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define epsilon 1e-6f

#define WGHT_SZ 25
#define WGHT_DIM_SZ 5
#define MAX_OFFSET 2

inline int mod(int x, int y)
{
    return (y + x%y)%y;
}

inline int calcLatticePos(int i, int j, int n)
{
    // Perform Modular Arithmetic
    return mod(i, n)*n + mod(j, n);  
}

////////////////////////////////////////////////////

void swapPtr(int** ptr1, int** ptr2)
{
    int *tempPtr = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tempPtr;
}

////////////////////////////////////////////////////

int calculateSpin(int *G, float *w, int n, int pos_i, int pos_j)
{
    float result = 0.0;
    for(int i=-MAX_OFFSET; i<=MAX_OFFSET; i++)
        for(int j=-MAX_OFFSET; j<=MAX_OFFSET; j++)
            result += w[ (i+MAX_OFFSET)*WGHT_DIM_SZ + (j+MAX_OFFSET) ]*G[ calcLatticePos(pos_i + i, pos_j + j, n) ];

    if( fabs(result) < epsilon)
        return G[pos_i*n + pos_j];  //doesn't change spin
    else if (result < 0)
        return -1;
    else
        return 1;
    
}

////////////////////////////////////////////////////

void ising(int *G, float *w, int k, int n)
{
    int *G2 = (int *)malloc((size_t)n*n*sizeof(int)); 

    for(int iter=0; iter<k; iter++)
    {
        for(int i=0; i<n; i++)
            for(int j=0; j<n; j++)
                G2[i*n + j] = calculateSpin(G, w, n, i, j);
        swapPtr(&G, &G2);   //G is always pointing to final struct
    }

    if(k%2 != 0)
    {
        /* Memory Address pointed by G does NOT change for caller.
         * So for odd number of iterations (k) we should swap
         * pointers and copy memory.
         */
        swapPtr(&G, &G2);
        memcpy(G, G2, (size_t)n*n*sizeof(int));
    }
    
    free(G2);
}