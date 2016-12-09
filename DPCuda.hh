#ifndef DP_CUDA_H
#define DP_CUDA_H

#include "Parallel-PTAS-4Oct-2016.hh"

void init_gpu(vector<DynamicTable> &AllTableElemets, vector<int> &counterVec);
void free_gpu(vector<DynamicTable> &AllTableElemets, vector<int> &counterVec);

__global__ void gpu_dpFunction(GpuDynamicTable *, int *, int, int);

#endif
