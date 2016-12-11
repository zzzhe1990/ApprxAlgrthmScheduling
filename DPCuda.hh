#ifndef DP_CUDA_H
#define DP_CUDA_H

#include "Parallel-PTAS-4Oct-2016.hh"

void free_gpu(vector<DynamicTable> &, vector<int> &, vector<int> &);
void init_gpu(vector<DynamicTable> &, vector<int> &, vector<int> &);
void call_gpu_dpFunction(int, int, int, int);


template<typename T>
__device__ T *gpu_realloc(int oldsize, int newsize, T *old);
template<typename T>
__device__ void gpu_push_back(T **array, T *elem, int *size);

__device__ void gpu_sumFun(int *, int * , int, int *, int);
__device__ void gpu_increase(const int *, int *, int, bool *);
__device__ void gpu_generate2(int *, int, thrust::device_vector<int> *, int*, thrust::device_vector<int> *,
															int, int, int *, thrust::device_vector<int>*);
__global__ void gpu_dpFunction(GpuDynamicTable *, int *, int, int);

#endif
