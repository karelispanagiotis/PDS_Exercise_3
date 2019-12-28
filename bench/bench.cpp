#include "ising.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

void generateLattice(int *G, int n)
{
    srand(time(NULL));
    for(int i=0; i<n*n; i++)
        if(rand()%2)    G[i] = 1;
        else    G[i] = -1;

}

int main(int arc, char** argv)
{
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);

    float w[25] = { 0.004,  0.016,  0.026,  0.016,   0.004,
                    0.016,  0.071,  0.117,  0.071,   0.016,
                    0.026,  0.117,  0.000,  0.117,   0.026,
                    0.016,  0.071,  0.117,  0.071,   0.016,
                    0.004,  0.016,  0.026,  0.016,   0.004 };
    
    int *G = (int *)malloc((size_t)n*n*sizeof(int));
    generateLattice(G, n);

    struct timeval start, end;
    long long int time_usec;
    double exec_time = 0;
 
    gettimeofday(&start, NULL);
    ising(G, w, k, n);
    gettimeofday(&end, NULL);

    time_usec = (end.tv_sec - start.tv_sec) * 1000000 
            + (end.tv_usec - start.tv_usec);
    exec_time = ((double)time_usec) / 1000000;

    char fileName[50];
    sprintf(fileName, "%s_log.txt", argv[0]);
    FILE* output = fopen(fileName, "a");
    fprintf(output,"%d,%d,%.5lf\n", n, k, exec_time);
    printf("%s: %dx%d for %d iter. took %lf\n\n", argv[0], n, n, k, exec_time);

    fclose(output);
    free(G);
    return 0;
}