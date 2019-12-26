#define Compilers
CC = gcc
CUDAC = nvcc

#define includes
INCLUDES = -I./inc/

#define flags
CFLAFS = -O3

#define object files to be made
OBJECTS = ising_sequential.o ising_cudaV1.o ising_cudaV2.o ising_cudaV3.o

lib: $(OBJECTS)

%.o: src/%.cpp
	$(CUDAC) $(INCLUDES) $(CFLAFS) -x cu -dc $< -o lib/$@