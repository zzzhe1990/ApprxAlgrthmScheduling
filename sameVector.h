#ifndef SAMEVEC_H
#define SAMEVEC_H
__device__ int gpu_sameVectors(int *vecA, int *vecB, int size);


__device__ int gpu_sameVectors(int *vecA, int choice, int size);


__global__ void gpu_sameVectors(int *A, int *B, const int powK, int *res);

#endif
