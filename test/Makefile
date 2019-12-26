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

tester_cudaV1: testerCuda.cpp
	cd ..; make lib;
	$(CUDAC) $(CFLAFS) $(INCLUDES) -x cu -dc $<
	$(CUDAC) $(CFLAFS) ../lib/ising_cudaV1.o testerCuda.o -o $@
	rm *.o

tester_cudaV2: testerCuda.cpp
	cd ..; make lib;
	$(CUDAC) $(CFLAFS) $(INCLUDES) -x cu -dc $<
	$(CUDAC) $(CFLAFS) ../lib/ising_cudaV2.o testerCuda.o -o $@
	rm *.o
	
sampleGen: sampleGen.cpp
	g++ $(INCLUDES) $< ../src/ising_sequential.cpp -o $@

clean:
	rm tester_* sampleGen