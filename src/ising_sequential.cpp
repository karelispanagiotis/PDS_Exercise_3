#include "ising.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define epsilon 1e-6f

inline int mod(int x, int y)
{
    return (y + x%y)%y;
}

inline int calcLatticePos(int pos, int n, int xOffset, int yOffset)
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

////////////////////////////////////////////////////

void swapPtr(int** ptr1, int** ptr2)
{
    int *tempPtr = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tempPtr;
}

////////////////////////////////////////////////////

int calculateSpin(int *G, float *w, int n, int pos)
{
    float result = 0.0;
    for(int i=-2; i<=2; i++)
        for(int j=-2; j<=2; j++)
            result += w[ (i+2)*5 + (j+2) ]*G[ calcLatticePos(pos, n, j, i) ];

    if( fabs(result) < epsilon)
        return G[pos];  //doesn't change spin
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
        for(int i=0; i<n*n; i++)
            G2[i] = calculateSpin(G, w, n, i);
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