all: run

run: DPCUDA.o
	
DPCUDA.o:
	nvcc DPCUDA.cu Parallel.cpp -g -arch=sm_35 -lcuda -lcudart -o  DPCUDA.o
