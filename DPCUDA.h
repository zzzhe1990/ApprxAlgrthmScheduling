#ifndef DPCUDA_H
#define	DPCUDA_H

#include "Parallel.h"
#include "iterator"

void InitGPUData(int powK, int LongJobs_size, vector<DynamicTable> &AllTableElemets, 
				 int *zeroVec, int *roundVec, int *counterVec, int &maxSubsetsSize, const int maxSumValue);
/*
void gpu_DP(vector<DynamicTable> &AllTableElemets, int *dev_ATE_elm, int *dev_counterVec, int *dev_roundVec, 
			const int T, const int k, const int powK, const int dev_AllTableElemets_size,
			int *dev_ATE_Csubsets, int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, 
			int Cwhole_size, int *dev_zeroVec, int *dev_ATE_optVector, int *dev_ATE_optVector_size,
			int *dev_ATE_myOPT, int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, 
			int *it, int *s, int *NS, const int maxSumValue, vector<int> &counterVec);
*/			
void gpu_DP(vector<DynamicTable> &AllTableElemets, const int T, const int k, const int powK, const int maxSumValue, 
			vector<int> &counterVec, const int LongJobs_size, int *zeroVec, int *roundVec);

/*
__global__ void FindOPT(int *dev_ATE_elm, int *dev_counterVec, int indexomp, int *dev_roundVec, 
						const int T, const int k, const int powK, const int dev_AllTableElemets_size,
						int *dev_ATE_Csubsets, int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, 
						int Cwhole_size, int *dev_zeroVec, int *dev_ATE_optVector, int *dev_ATE_optVector_size,
						int *dev_ATE_myOPT, int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, 
						const int i, int *it, int *s, int *NS);
						
__device__ void gpu_generate2(int *Ntemp, const int Ntemp_size, int *Ctemp, int *NMinusStemp, int *dev_roundVec, 
							  const int T, const int powk, int *it, int *s, int *NS);

__device__ int gpu_sameVectors(int *vecA, int *vecB, int size);
__device__ int gpu_increase(const int *Ntemp, int *it, int Ntemp_size);
__device__ int gpu_sumFun(int *A, int *B, const int powK);
*/

#endif
