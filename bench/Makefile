#define Compilers
CUDAC = nvcc

#define includes
INCLUDES = -I../inc/

#define flags
CFLAFS = -O3

benchmark_sequential: bench.cpp
	cd ..; make lib;
	$(CUDAC) $(CFLAFS) $(INCLUDES) -c $<
	$(CUDAC) $(CFLAFS) ../lib/ising_sequential.o bench.o -o $@
	rm *.o

benchmark_cudaV1: bench.cpp
	cd ..; make lib;
	$(CUDAC) $(CFLAFS) $(INCLUDES) -x cu -dc $<
	$(CUDAC) $(CFLAFS) ../lib/ising_cudaV1.o bench.o -o $@
	rm *.o

benchmark_cudaV2: bench.cpp
	cd ..; make lib;
	$(CUDAC) $(CFLAFS) $(INCLUDES) -x cu -dc $<
	$(CUDAC) $(CFLAFS) ../lib/ising_cudaV2.o bench.o -o $@
	rm *.o

benchmark_cudaV3: bench.cpp
	cd ..; make lib;
	$(CUDAC) $(CFLAFS) $(INCLUDES) -x cu -dc $<
	$(CUDAC) $(CFLAFS) ../lib/ising_cudaV3.o bench.o -o $@
	rm *.o

clean:
	rm benchmark_sequential benchmark_cudaV1 benchmark_cudaV2 benchmark_cudaV3