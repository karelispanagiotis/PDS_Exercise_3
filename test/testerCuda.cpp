#include "ising.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    char file[50];

    int *G = (int *)malloc(n*n*sizeof(int));
    int *result = (int *)malloc(n*n*sizeof(int));

    sprintf(file, "samples/%d/conf-init.bin", n);
    FILE *initialState = fopen(file, "rb");
    sprintf(file, "samples/%d/conf-%d.bin", n, k);
    FILE *finalState = fopen(file, "rb");

    size_t read;
    read = fread(G, sizeof(int), n*n, initialState);
    read = fread(result, sizeof(int), n*n, finalState);
    if(read != n*n)
    {
        printf("Reading Error!\n Aborting...\n");
        exit(0);
    }

    float w[25] = { 0.004,  0.016,  0.026,  0.016,   0.004,
                    0.016,  0.071,  0.117,  0.071,   0.016,
                    0.026,  0.117,  0.000,  0.117,   0.026,
                    0.016,  0.071,  0.117,  0.071,   0.016,
                    0.004,  0.016,  0.026,  0.016,   0.004 };
    
    ising(G, w, k, n);
 
    int isCorrect = 0;
    for(int i=0; i<n*n; i++)
        if(G[i] != result[i])
        {
            isCorrect = 1;
            break;
        }

    if(isCorrect == 0)
        printf("Correct for n = %d, k = %d\n", n, k);
    else   
        printf("WRONG! for n = %d, k = %d\n", n, k);
    return 0;
}