#include "ising.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
    int n = 517;
    int k = atoi(argv[1]);
    // Define Weight Matrix
    float w[25] = { 0.004,  0.016,  0.026,  0.016,   0.004,
                    0.016,  0.071,  0.117,  0.071,   0.016,
                    0.026,  0.117,  0.000,  0.117,   0.026,
                    0.016,  0.071,  0.117,  0.071,   0.016,
                    0.004,  0.016,  0.026,  0.016,   0.004 };

    // Define Ising Matrix
    int *G = (int *) malloc((size_t)n*n*sizeof(int));

    FILE *input = fopen("conf-init.bin", "rb");
    FILE *output = fopen("output.bin", "wb");

    fread(G, sizeof(int), n*n, input);

    ising(G, w, k, n);

    fwrite(G, sizeof(int), n*n, output);
    fclose(output);
    fclose(input);

    char command[50];
    sprintf(command, "diff output.bin conf-%d.bin", k);
    int rv = system(command);
    if(rv == 0)
        printf("CORRECT\n");
    else
        printf("WRONG!\n");

    return 0;
}