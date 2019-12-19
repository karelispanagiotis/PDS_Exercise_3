#include "ising.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define epsilon 1e-6

int calculatePos(int pos, int n, int xOffset, int yOffset)
{
    // Find coordinates
    int i = pos/n;
    int j = pos%n;
    // Perform modular arithmetic for the result
    return ((n + (i + yOffset))%n)*n + (n + (j + xOffset))%n;  
}

////////////////////////////////////////////////////

void swapPtr(int** ptr1, int** ptr2)
{
    int *tempPtr = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tempPtr;
}

////////////////////////////////////////////////////

int calculateSpin(int *G, double *w, int n, int pos)
{
    double result = 0.0;
    for(int i=-2; i<=2; i++)
        for(int j=-2; j<=2; j++)
        {
            result += w[ (i+2)*5 + (j+2) ]*G[ calculatePos(pos, n, j, i) ];
        }

    if( fabs(result) < epsilon)
        return G[pos];  //doesn't change spin
    else if (result < 0)
        return -1;
    else
        return 1;
    
}

void ising(int *G, double *w, int k, int n)
{
    int *G2 = malloc((size_t)n*n*sizeof(int)); 

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