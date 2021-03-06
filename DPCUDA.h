#ifndef DPCUDA_H
#define	DPCUDA_H

int *dev_ATE_elm, *dev_ATE_myOPT, *dev_ATE_myOptimalindex, *dev_ATE_myMinNSVector;
int *dev_ATE_NSsubsets, *dev_ATE_Csubsets;
int *dev_ATE_optVector;
int *dev_counterVec;
int *dev_ATE_elm_size;
int *dev_ATE_NSsubsets_size;
int *dev_ATE_optVector_size;
int *dev_zeroVec, *dev_roundVec;
int *dev_ATE_NSsubsets_size, *dev_ATE_optVector_size;


__global__ void FindOPT(int *dev_ATE_elm, int *dev_counterVec, int indexomp, int *dev_roundVec, const int T, const int k, const int powK, const int dev_AllTableElemets_size,
						int *dev_ATE_Csubsets, int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, int Cwhole_size, int *dev_zeroVec, int *dev_ATE_optVector, int *dev_ATE_optVector_size,
						int *dev_ATE_myOPT, int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector);
						
__device__ void gpu_generate2(int *Ntemp, int Ntemp_size, int *Ctemp, int *NMinusStemp, int *dev_roundVec, const int T, const int powk);

__device__ int gpu_sameVectors(int *vecA, int *vecB, int size);
__device__ int gpu_increase(const int *Ntemp, int *it, int Ntemp_size);
__device__ int gpu_sumFun(int *A, int *B);


#endif
