#ifndef DPCUDA_H
#define	DPCUDA_H

#include "Parallel.h"
#include "iterator"


//#define DEBUG

//void InitGPUData(int powK, int LongJobs_size, vector<DynamicTable> &AllTableElemets, 
//				 int *zeroVec, int *roundVec, int *counterVec, int &maxSubsetsSize, const int optVectorSize);
void InitGPUData(int powK, int LongJobs_size, vector<DynamicTable> &AllTableElemets, int *zeroVec, 
				 int *roundVec, int *counterVec, int &maxSubsetsSize, const int optVectorSize, 
				 const int counterVecSize, int **dev_ATE_elm, int **dev_ATE_myOPT, int **dev_ATE_myOptimalindex, 
				 int **dev_ATE_myMinNSVector, int **dev_ATE_NSsubsets, int **dev_ATE_Csubsets, int **dev_ATE_optVector,
				 int **dev_counterVec, int **dev_ATE_NSsubsets_size, int **dev_ATE_optVector_size, int **dev_zeroVec, 
				 int **dev_roundVec, int **it, int **ss, int **NS);


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

void gpu_BlockDP(vector<DynamicTable> &AllTableElemets, const int T, const int powK, const int jobsPerBlock, const int levelsPerBlock,
			vector<int> &counterVec, const int LongJobs_size, int *zeroVec, int *roundVec, vector<int> &divisor, vector<int> &divisorComp,
			vector<int> &blockDimSize, vector<block> &allBlocks, vector<block> &allBlocksNoZero, vector<int> &blockCounterVec);

#endif
