#include "ising.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* For a given argument n,
 * this program generates binary files for:
 *  *Initial state  : conf-init.bin
 *  *k-th step state: conf-k.bin
 * where k = 1, 3, 5, 8
 * 
 * Each file is stored inside the directory 
 */
void generateLattice(int *G, int n)
{
    srand(time(NULL));
    for(int i=0; i<n*n; i++)
        if(rand()%2)    G[i] = 1;
        else    G[i] = -1;

}
int main(int argc, char **argv)
{
    int n = atoi(argv[1]);
    char dir[128];
    sprintf(dir, "samples/%d/", n);
    FILE *output;
    
    //Generate Random Initial Lattice
    int *G = (int *)malloc((size_t)n*n*sizeof(int));
    generateLattice(G, n);

    float w[25] = { 0.004,  0.016,  0.026,  0.016,   0.004,
                    0.016,  0.071,  0.117,  0.071,   0.016,
                    0.026,  0.117,  0.000,  0.117,   0.026,
                    0.016,  0.071,  0.117,  0.071,   0.016,
                    0.004,  0.016,  0.026,  0.016,   0.004 };
    
    output = fopen(strcat(dir, "conf-init.bin"), "wb");
    fwrite(G, sizeof(int), n*n, output);
    fclose(output);

    //Perfrom Ising
    ising(G, w, 1, n);  //k = 1;
    sprintf(dir, "samples/%d/", n);
    output = fopen(strcat(dir, "conf-1.bin"), "wb");
    fwrite(G, sizeof(int), n*n, output);
    fclose(output);

    ising(G, w, 2, n);  //k = 3;
    sprintf(dir, "samples/%d/", n);
    output = fopen(strcat(dir, "conf-3.bin"), "wb");
    fwrite(G, sizeof(int), n*n, output);
    fclose(output);

    ising(G, w, 2, n);  //k = 5;
    sprintf(dir, "samples/%d/", n);
    output = fopen(strcat(dir, "conf-5.bin"), "wb");
    fwrite(G, sizeof(int), n*n, output);
    fclose(output);

    ising(G, w, 3, n);  //k = 8;
    sprintf(dir, "samples/%d/", n);
    output = fopen(strcat(dir, "conf-8.bin"), "wb");
    fwrite(G, sizeof(int), n*n, output);
    fclose(output);

    return 0;
}