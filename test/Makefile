#define Compilers
CUDAC = nvcc

#define includes
INCLUDES = -I../inc/

#define flags
CFLAFS = -O3

tester_sequential: tester.cpp
	cd ..; make lib;
	$(CUDAC) $(CFLAFS) $(INCLUDES) -c $<
	$(CUDAC) $(CFLAFS) ../lib/ising_sequential.o tester.o -o $@
	rm *.o

tester_cudaV1: tester.cpp
	cd ..; make lib;
	$(CUDAC) $(CFLAFS) $(INCLUDES) -x cu -dc $<
	$(CUDAC) $(CFLAFS) ../lib/ising_cudaV1.o tester.o -o $@
	rm *.o