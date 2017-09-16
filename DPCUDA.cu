#include "DPCUDA.h"
#include "sameVector.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


const int maxStreamNum = 64;
cudaStream_t streams[maxStreamNum];


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
void InitGPUData(int powK, int LongJobs_size, vector<DynamicTable> &AllTableElemets, int *zeroVec,
				 int *roundVec, int *counterVec, int &maxSubsetsSize, const int optVectorSize,
				 const int counterVecSize, int **dev_ATE_elm, int **dev_ATE_myOPT, int **dev_ATE_myOptimalindex,
				 int **dev_ATE_myMinNSVector, int **dev_ATE_NSsubsets, int **dev_ATE_Csubsets, int **dev_ATE_optVector,
				 int **dev_counterVec, int **dev_ATE_NSsubsets_size, int **dev_ATE_optVector_size, int **dev_zeroVec,
				 int **dev_roundVec, int **it, int **ss, int **NS)
{
	//cout << "Beginning of InitGPUData, thread: " << omp_get_thread_num() << ", dev_roundVec address: " << *dev_roundVec << endl;

	int maxIndex = AllTableElemets.size() - 1;
	int maxCounterVec = 0;
	vector<int> temp;
//	for (vector<int>::const_iterator pt = AllTableElemets[maxIndex].elm.end(); pt != AllTableElemets[maxIndex].elm.begin(); --pt)
	for (int p = powK-1; p >= 0; p--)
	{
		if (AllTableElemets[maxIndex].elm[p] != 0)
			temp.push_back(AllTableElemets[maxIndex].elm[p]);
	}

	for (int i = 0; i < temp.size(); i++)
	{
		int a = 1;
		for (int j = 0; j < i + 1; j++)
		{
			a *= temp[j];
		}
		cout << "Update maxSubsetsSize, current: " << maxSubsetsSize << ", a: " << a <<endl;
		maxSubsetsSize += a;
	}

	for (int i = 0; i < counterVecSize; i++)
	{
		if (counterVec[i] > maxCounterVec)
			maxCounterVec = counterVec[i];
	}

	cout << "AllTableSize: " << AllTableElemets.size() << ", maxSubsetsSize: " << maxSubsetsSize << ", powK: " << powK << ", tempSize: " << temp.size() << endl;

	//arrays on device
	gpuErrchk(cudaMalloc((void***)&dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&dev_ATE_elm, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&dev_ATE_myMinNSVector, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&dev_ATE_myOPT, AllTableElemets.size() * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&dev_ATE_optVector, AllTableElemets.size() * optVectorSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&dev_counterVec, (LongJobs_size + 1) * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&dev_zeroVec, (powK) * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&dev_roundVec, (powK) * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&it, maxCounterVec * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&ss, maxCounterVec * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&NS, maxCounterVec * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void***)&dev_ATE_NSsubsets_size, AllTableElemets.size() * sizeof(int)));
    gpuErrchk(cudaMalloc((void***)&dev_ATE_optVector_size, AllTableElemets.size() * sizeof(int)));


	//cout << "thread: " << omp_get_thread_num() << ", cuda initial is done." << " Start memcpy! AllTableElemets.size: " << AllTableElemets.size() << endl;


	int *ATE_myOPT = new int[AllTableElemets.size()];

	for (int i = 0; i < AllTableElemets.size(); i++){
		gpuErrchk(cudaMemcpyAsync(&dev_ATE_elm[i * powK], &AllTableElemets[i].elm[0], powK * sizeof(int), cudaMemcpyHostToDevice, 0));
		ATE_myOPT[i] = AllTableElemets[i].myOPT;
	}

	//gpuErrchk(cudaMemcpyAsync(dev_zeroVec, zeroVec, powK*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(*dev_zeroVec, 0, powK*sizeof(int)));
	gpuErrchk(cudaMemcpy(*dev_roundVec, roundVec, powK*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*dev_counterVec, counterVec, (LongJobs_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(*dev_ATE_myOPT, ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyHostToDevice));

	delete[] ATE_myOPT;
	//cout << "End of InitGPUData, thread: " << omp_get_thread_num() << ", dev_roundVec address: " << *dev_roundVec << endl;
}
*/
/*
__device__ int gpu_sameVectors(int *vecA, int *vecB, int size)
{
	int same = 1;
	for (int i = 0; i < size; i++)
	{
		if (vecA[i] != vecB[i])
		{
			same = 0;
			break;
		}
	}
	return same;
}

__device__ int gpu_sameVectors(int *vecA, int choice, int size)
{
	int vecB[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0};

	vecB[15] = choice;

	int same = 1;
	for (int i = 0; i < size; i++)
	{
		if (vecA[i] != vecB[i])
		{
			same = 0;
			break;
		}
	}
	return same;
}


__global__ void gpu_sameVectors(int *A, int *B, const int powK, int *res)
{
	int thread = blockDim.x * blockIdx.x + threadIdx.x;
	int tRes;
//	__shared__ int warpRes[32];

//	if (thread < 32)
//		warpRes[thread] = 0;

	if (thread < powK)
	{
		tRes = __all(A[thread]-B[thread]);
	}

//	if (thread&(32-1) == 0)
//	{
//		warpRes[thread/32] = tRes;
//	}

//	if (thread < 32)
//		tRes = __any( warpRes != 0 );

	if (thread == 0)
		res[0] = tRes;
}
*/

__device__ int gpu_increase(int *Ntemp, int *it, int Ntemp_size)
{
	int index;
	for (int i = 0; i < Ntemp_size; i++)
	{
		index = Ntemp_size - 1 - i;
		++it[index];
		if (it[index] > Ntemp[index]) {
			it[index] = 0;
		}
		else {
			return 1;
		}
	}
	return 0;
}


__device__ int gpu_sumFun(int *A, int *B, const int powK)
{
	int summ=0.0;
//#pragma unroll
	for(int i=0; i<powK; i++)
	{
		summ= summ + A[i]*B[i];
	}
	return summ;
}

__host__ __device__ int gpu_blockOffsetNoZero(int *block, int *div, int blockSize, int divSize)
{
	int blockOffset = 0;
	for (int i = 0; i < blockSize; i++)
	{
		int offset = block[i];
		for (int j = i+1 ; j < divSize; j++)
			offset *= div[j];
		blockOffset += offset;
	}
	return blockOffset;
}

__device__ int gpu_blockOffset(int *block, int *divComp, int blockSize, int *div, int divSize)
{
	int blockOffset = 0, count = 0;
	for (int i = 0; i < blockSize; i++)
	{
		if (divComp[i] != 0)
		{
			int offset = block[i];
			for (int j = count+1 ; j < divSize; j++)
				offset *= div[j];
			blockOffset += offset;
			count++;
		}
	}
	return blockOffset;
}


//FOr the selecting current configure Nsub, find all its sub-configures and update corresponding C set and NSsub set
__global__ void gpu_genSubConfigs(int *jNSub, int *jC, int *jNSsubsets, int *jConfigSize, const int jNSize,
								  const int totalThread, const int powK, const int T, int *jCountC,
								  int *jCountNS, int *dev_roundVec)
{
	int thread = blockDim.x * blockIdx.x + threadIdx.x;

	//this is the maximum # of dimensions that a configuration can have

	if (thread < totalThread)
	{
		int subConfig[64];
		int NS[64];
		int remain = thread;
		int index = 0;
		//For a given thread, find its corresponding sub-configure and store into subConfig at first jNsize pos.
		for (int i=0; i < powK; i++)
		{
			if (jNSub[i] == 0)
			{
				subConfig[i] = 0;
			}
			else
			{
				int offset = 1;
				for (int j = index+1; j < jNSize; j++)
					offset *= jConfigSize[j];

				subConfig[i] = remain / offset;
				remain -= (subConfig[i] * offset);
				index++;
			}
		}

		int sSum=gpu_sumFun(&subConfig[0], dev_roundVec, powK);

		if(sSum <= T)
		{
			jCountC[thread] = 1;
			for (int i = 0; i < powK; i++)
			{
				jC[thread*powK + i] = subConfig[i];
				NS[i] = jNSub[i] - subConfig[i];
			}

			if(gpu_sameVectors(&NS[0], jNSub, powK) == 0)
			{
				jCountNS[thread] = 1;
				for (int i = 0; i < powK; i++)
				{
					jNSsubsets[thread * powK + i] = NS[i];
				}
			}
		}
	}
}

/*
__global__ void gpu_generate2(const int maxSubsetsSize, int *dev_ATE_elm, const int powK,
							  int *dev_ATE_Csubsets, int *dev_ATE_NSsubsets, int *dev_roundVec,
							  const int T, int *dev_ATE_NSsubsets_size,
							  int *dev_counterVec, const int ii, const int indexomp,
							  int *dev_subConfigSize, int *dev_countC, int *dev_countNS)
{
	int thread = blockDim.x * blockIdx.x + threadIdx.x;
	int j = thread + indexomp;

	if (thread < dev_counterVec[ii]){
		int *jN = &dev_ATE_elm[j * powK];
		int *jC = &dev_ATE_Csubsets[j * maxSubsetsSize * powK];
		int *jNSsubsets = &dev_ATE_NSsubsets[j * maxSubsetsSize * powK];
		int *jCountC = &dev_countC[j * maxSubsetsSize];
		int *jCountNS = &dev_countNS[j * maxSubsetsSize];
		int *jConfigSize = &dev_subConfigSize[thread * powK];

		int jNSize = 0;
		int numThread = 1;

		for (int i=0; i<powK; i++)
		{
			if (jN[i] != 0)
			{
				jConfigSize[jNSize] = jN[i] + 1;
				jNSize++;
				numThread *= (jN[i]+1);
			}
		}
		//cudaDeviceSynchronize();

		int threadsPerBlock = 64;
		int blocksPerGrid = (numThread + threadsPerBlock -1) / threadsPerBlock;

		gpu_genSubConfigs<<<blocksPerGrid, threadsPerBlock>>>(jN, jC, jNSsubsets, jConfigSize, jNSize,
															numThread, powK, T, jCountC, jCountNS, dev_roundVec);
	}
}
*/

__global__ void FindAllSub(const int powK, int *jN, int *dev_roundVec, const int T, int *jNSsubsets, 
						   int *dev_lock, int *dev_ATE_NSsubsets_size, const int id, const int allSubSize,
						   int cpuId, int blockLvl, int blockIDInLvl)
{
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (thread < allSubSize){		
		int js[16];
		int jNS[16];
		
		int residual = thread;
		for (int i = 0; i < powK; i++)
		{
			js[i] = 0;
			int div = 1;
			if (jN[i] != 0){				
				for (int j=i+1; j<powK; j++){
					if (jN[j] != 0){
						div *= (jN[j] + 1);
					}
				}
				js[i] = residual / div;
				residual -= (div * js[i]);
			}
		}	

		int sSum=gpu_sumFun(js, dev_roundVec, powK);

		if(sSum <= T)
		{		
			for(int i=0; i<powK; i++)
			{
				jNS[i] = jN[i] - js[i];
			}
			
			if(gpu_sameVectors(jNS, jN, powK) == 0){	
				bool leave = true;
				while (leave) {
					if (atomicCAS(&dev_lock[id], 0, 1) == 0) {
						for (int i = 0; i < powK; i++)
						{
							jNSsubsets[dev_ATE_NSsubsets_size[id] * powK + i] = jNS[i];
						}
						atomicAdd(&dev_ATE_NSsubsets_size[id], 1);
						leave = false;
						
						atomicExch(&dev_lock[id], 0);
					}
				}
			}
		}
	}
}

/*
__global__ void gpu_generate2(const int maxSubsetsSize, int *dev_ATE_elm, const int powK,
							  int *dev_ATE_Csubsets, int *dev_ATE_NSsubsets, int *dev_roundVec,
							  const int T, int *it, int *s, int *NS, int *dev_ATE_NSsubsets_size,
							  int *dev_counterVec, const int ii, const int indexomp, int *dev_lock,
							  int cpuId, int blockLvl, int blockIDInLvl)
*/
__global__ void gpu_generate2(const int maxSubsetsSize, int *dev_ATE_elm, const int powK,
							  int *dev_ATE_NSsubsets, int *dev_roundVec, const int T, 
							  int *dev_ATE_NSsubsets_size, int *dev_counterVec, 
							  const int ii, const int indexomp, int *dev_lock, int cpuId, 
							  int blockLvl, int blockIDInLvl)
{
	int thread = blockDim.x * blockIdx.x + threadIdx.x;
	int j = thread + indexomp;

	if (thread < dev_counterVec[ii]){
		int *jN = &dev_ATE_elm[j * powK];
		//int *jC = &dev_ATE_Csubsets[j * maxSubsetsSize * powK];
		int *jNSsubsets = &dev_ATE_NSsubsets[j * maxSubsetsSize * powK];
		//int *jit = &it[thread * powK];
		//int *js = &s[thread * powK];
		//int *jNS = &NS[thread * powK];

		//int counterNS = 0, counterC = 0;
		int allSubSize = 1;

		for (int i = 0; i < powK; i++)
		{
			//jit[i] = 0;
			//js[i] = 0;
			//jNS[i] = 0;
			allSubSize *= (jN[i]+1);
		}
/*
//		if (allSubSize < 16){
			do {
				for (int i = 0; i < powK; i++)
				{
					js[i] = jit[i];
				}
									
				int sSum=gpu_sumFun(js, dev_roundVec, powK);

				if(sSum <= T)
				{
					for (int i = 0; i < powK; i++)
					{
						jC[counterC*powK + i] = js[i];
					}
										
					counterC++;		
					
					for(int i=0; i<powK; i++)
					{
						jNS[i] = jN[i] - js[i];
					}

					if(gpu_sameVectors(jNS, jN, powK)){
						continue;
					}

					for (int i = 0; i < powK; i++)
					{
						jNSsubsets[counterNS * powK + i] = jNS[i];
					}

					counterNS++;
				}
			}while (gpu_increase(jN, jit, powK));	
			dev_ATE_NSsubsets_size[j] = counterNS;
/*		}
		else{
*/			int blockSize = 32;
			int gridSize = (allSubSize + blockSize - 1) / blockSize;
			FindAllSub<<<gridSize, blockSize>>>(powK, jN, dev_roundVec, T, jNSsubsets, dev_lock, dev_ATE_NSsubsets_size, j, allSubSize, cpuId, blockLvl, blockIDInLvl);
//		}*/
	}
}

__global__ void FindSubConfigOPT(int MemOffset, int *NS, int *dev_ATE_elm, int powK, 
								 int *dev_ATE_optVector, int optOffset, int optVecIndex, 
								 int *dev_ATE_optVector_size, int j, int *dev_ATE_myOPT,
								 int jobsPerBlock)
{
	__shared__ int lock[1];
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	int r = MemOffset + thread;
	if (thread == 0)
		lock[1] = 0;
	__syncthreads();
	
	if (thread < jobsPerBlock){
		if (lock[1] == 0){
			if (gpu_sameVectors(NS, &dev_ATE_elm[r * powK], powK))
			{
				dev_ATE_optVector[optOffset + optVecIndex] = dev_ATE_myOPT[r];
				dev_ATE_optVector_size[j] = optVecIndex+1;
				atomicAdd(lock, 1);
			}
		}
	}
}

/*
__global__ void LoopNSsubsets(int *dev_ATE_NSsubsets_size, int *dev_ATE_NSsubsets, int NSOffset,
							  int powK, int *dev_zeroVec, int *blockDimSize, int *divisorComp, int *divisor, 
							  int divSize, int jobsPerBlock, int *dev_ATE_elm, volatile int *dev_ATE_myOPT,
							  const int j, const int cpuId, const int configOffset, int *dev_ATE_myMinNSVector,
							  int *dev_ATE_myOptimalindex, const int optOffset, int *dev_ATE_optVector)
*/
__global__ void LoopNSsubsets(int *dev_ATE_NSsubsets_size, int *dev_ATE_NSsubsets, int NSOffset,
							  int powK, int *dev_zeroVec, int *blockDimSize, int *divisorComp, int *divisor, 
							  int divSize, int jobsPerBlock, int *dev_ATE_elm, volatile int *dev_ATE_myOPT,
							  const int j, const int cpuId, const int configOffset, int *dev_ATE_myMinNSVector,
							  int *dev_ATE_myOptimalindex)
{
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	int *NS = &dev_ATE_NSsubsets[(NSOffset + thread) * powK];
		
	__shared__ int lock[1];
	__shared__ int opt[64];
	__shared__ int idx[64];
	
	//dev_ATE_optVector[optOffset + thread] = 1000000;
	opt[thread] = 100000;
	idx[thread] = thread;
	
	if (thread == 0)
		lock[0] = 0;
	__syncthreads();
	
	if (thread < dev_ATE_NSsubsets_size[j]){
		
		if(gpu_sameVectors(NS, dev_zeroVec, powK))
		{
			//dev_ATE_optVector[optOffset + thread] = 0;
			opt[thread] = 0;
			lock[0] = 1;
		}
	}
	__syncthreads();
	
	if (thread < dev_ATE_NSsubsets_size[j]){
		int blockIndex[32];
		
		if (lock[0] == 0){
			//Find in which block the sub-configuration is stored.
			for (int i = 0; i < powK; i++)
			{
				if (blockDimSize[i] != 0)
				{
					blockIndex[i] = NS[i] / blockDimSize[i];
				}
			}

			int blockOffset = gpu_blockOffset(&blockIndex[0], divisorComp, powK, divisor, divSize);
			int MemOffset = blockOffset * jobsPerBlock;
		
			for (int r = MemOffset; r < MemOffset + jobsPerBlock; r++)
			{	
				if (gpu_sameVectors(NS, &dev_ATE_elm[r * powK], powK))
				{
					//dev_ATE_optVector[optOffset + thread] = dev_ATE_myOPT[r];
					opt[thread] = dev_ATE_myOPT[r];
					break;
				}
			} 
		}
	}
	__syncthreads();
			
	if (thread < dev_ATE_NSsubsets_size[j]){
/*		if (thread > 32){
			if (dev_ATE_optVector[optOffset + thread] < dev_ATE_optVector[optOffset + thread - 32]){
				dev_ATE_optVector[optOffset + thread - 32] = dev_ATE_optVector[optOffset + thread];
				idx[thread-32] = idx[thread];
			}
		}
		__syncthreads();
		if (thread > 16 && thread <= 32 ){
			if (dev_ATE_optVector[optOffset + thread] < dev_ATE_optVector[optOffset + thread - 16]){
				dev_ATE_optVector[optOffset + thread - 16] = dev_ATE_optVector[optOffset + thread];
				idx[thread-16] = idx[thread];
			}
		}			
		__syncthreads();
		if (thread > 8 && thread <= 16 ){
			if (dev_ATE_optVector[optOffset + thread] < dev_ATE_optVector[optOffset + thread - 8]){
				dev_ATE_optVector[optOffset + thread - 8] = dev_ATE_optVector[optOffset + thread];
				idx[thread-8] = idx[thread];
			}
		}			
		__syncthreads();
		if (thread > 4 && thread <= 8 ){
			if (dev_ATE_optVector[optOffset + thread] < dev_ATE_optVector[optOffset + thread - 4]){
				dev_ATE_optVector[optOffset + thread - 4] = dev_ATE_optVector[optOffset + thread];
				idx[thread-4] = idx[thread];
			}
		}			
		__syncthreads();
		if (thread > 2 && thread <= 4 ){
			if (dev_ATE_optVector[optOffset + thread] < dev_ATE_optVector[optOffset + thread - 2]){
				dev_ATE_optVector[optOffset + thread - 2] = dev_ATE_optVector[optOffset + thread];
				idx[thread-2] = idx[thread];
			}
		}			
		__syncthreads();
		
		if (thread == 0){
			if (dev_ATE_optVector[optOffset + 1] < dev_ATE_optVector[optOffset]){
				dev_ATE_optVector[optOffset] = dev_ATE_optVector[optOffset + 1];
				idx[0] = idx[1];
			}
			
			dev_ATE_myOPT[configOffset + j] = dev_ATE_optVector[optOffset] + 1;
			dev_ATE_myOptimalindex[configOffset + j] = idx[0];
		}
		__syncthreads();
*/		
		
		if (thread >= 32){
			if (opt[thread] < opt[thread - 32]){
				opt[thread - 32] = opt[thread];
				idx[thread-32] = idx[thread];
			}
		}
		__syncthreads();
		if (thread >= 16 && thread < 32 ){
			if (opt[thread] < opt[thread - 16]){
				opt[thread - 16] = opt[thread];
				idx[thread-16] = idx[thread];
			}
		}			
		__syncthreads();
		if (thread >= 8 && thread < 16 ){
			if (opt[thread] < opt[thread - 8]){
				opt[thread - 8] = opt[thread];
				idx[thread-8] = idx[thread];
			}
		}			
		__syncthreads();
		if (thread >= 4 && thread < 8 ){
			if (opt[thread] < opt[thread - 4]){
				opt[thread - 4] = opt[thread];
				idx[thread-4] = idx[thread];
			}
		}			
		__syncthreads();
		if (thread >= 2 && thread < 4 ){
			if (opt[thread] < opt[thread - 2]){
				opt[thread - 2] = opt[thread];
				idx[thread-2] = idx[thread];
			}
		}			
		__syncthreads();
		
		if (thread == 0){
			if (opt[1] < opt[0]){
				opt[0] = opt[1];
				idx[0] = idx[1];
			}
			
			dev_ATE_myOPT[configOffset + j] = opt[0] + 1;
			dev_ATE_myOptimalindex[configOffset + j] = idx[0];
		}
		__syncthreads();
	
		if (thread < powK){
			dev_ATE_myMinNSVector[(j + configOffset) * powK + thread] = dev_ATE_NSsubsets[(NSOffset + idx[0]) * powK + thread];
		}
	}
}

#ifdef SPLIT
/*
__global__ void FindOPT(int *dev_ATE_elm, int *dev_counterVec, int indexomp, int *dev_roundVec,
						const int powK, const int AllTableElemets_size, int *dev_ATE_Csubsets,
						int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, int *dev_zeroVec,
						int *dev_ATE_myOPT, int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, 
						const int ii, const int maxSubsetsSize, int *blockDimSize, int *divisor, 
						int *divisorComp, const int divSize, const int jobsPerBlock, const int cpuId, 
						const int configOffset, const int blockLvl, const int blockIDInLvl,
						int *dev_ATE_optVector, const int optVectorSize)
*/
__global__ void FindOPT(int *dev_ATE_elm, int *dev_counterVec, int indexomp, int *dev_roundVec,
						const int powK, const int AllTableElemets_size, int *dev_ATE_NSsubsets, 
						int *dev_ATE_NSsubsets_size, int *dev_zeroVec, int *dev_ATE_myOPT, 
						int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, const int ii, 
						const int maxSubsetsSize, int *blockDimSize, int *divisor, int *divisorComp, 
						const int divSize, const int jobsPerBlock, const int cpuId, const int configOffset, 
						const int blockLvl, const int blockIDInLvl)
{
	int thread = blockDim.x * blockIdx.x + threadIdx.x;

	int j = thread + indexomp;
	int NSOffset = j * maxSubsetsSize;
	//int optOffset = j * optVectorSize;

	if (thread < dev_counterVec[ii]){
		
		int blockSize = 64;
		int gridSize = (dev_ATE_NSsubsets_size[j] + blockSize - 1) / blockSize;
/*				
		LoopNSsubsets<<<gridSize, blockSize>>>(dev_ATE_NSsubsets_size, dev_ATE_NSsubsets, NSOffset, 
												powK, dev_zeroVec, blockDimSize, divisorComp, divisor, 
												divSize, jobsPerBlock, dev_ATE_elm, dev_ATE_myOPT, j, 
												cpuId, configOffset, dev_ATE_myMinNSVector, 
												dev_ATE_myOptimalindex, optOffset, dev_ATE_optVector);
*/
		LoopNSsubsets<<<gridSize, blockSize>>>(dev_ATE_NSsubsets_size, dev_ATE_NSsubsets, NSOffset, 
												powK, dev_zeroVec, blockDimSize, divisorComp, divisor, 
												divSize, jobsPerBlock, dev_ATE_elm, dev_ATE_myOPT, j, 
												cpuId, configOffset, dev_ATE_myMinNSVector, dev_ATE_myOptimalindex);
	}
/*
		int blockIndex[64];
		for(int h=0; h < dev_ATE_NSsubsets_size[j]; h++)
		{
			//actually, h == optVectorIndex, and optVector_size = optVecIndex+1;			
			int *NS = &dev_ATE_NSsubsets[(NSOffset + h) * powK];
			if(gpu_sameVectors(NS, dev_zeroVec, powK))
			{
				dev_ATE_optVector[optOffset + h] = 0;
				dev_ATE_optVector_size[j] = h+1;
				break;
			}

			//Find in which block the sub-configuration is stored.
			for (int i = 0; i < powK; i++)
			{
				if (blockDimSize[i] != 0)
				{
					blockIndex[i] = NS[i] / blockDimSize[i];
				}
			}

			int blockOffset = gpu_blockOffset(&blockIndex[0], divisorComp, powK, divisor, divSize);
			int MemOffset = blockOffset * jobsPerBlock;
			bool nomatch = true;
			for (int r = MemOffset; r < MemOffset + jobsPerBlock && nomatch; r++)
			{	
				if (gpu_sameVectors(NS, &dev_ATE_elm[r * powK], powK))
				{
					dev_ATE_optVector[optOffset + h] = dev_ATE_myOPT[r];
					dev_ATE_optVector_size[j] = h+1;
					nomatch = false;
				}
			} 
		}

	if (thread < dev_counterVec[ii]){
		int minn = 100000;
		int myOptimalindex = 0;
		// find out the OPT from all dependencies.		
		for (int pp = 0; pp < dev_ATE_optVector_size[j]; pp++)
		{			
			if (dev_ATE_optVector[optOffset + pp] < minn)
			{
				minn = dev_ATE_optVector[optOffset + pp];
				myOptimalindex=pp;
			}
		}
		
		int optTemp=minn+1;
		dev_ATE_myOPT[configOffset + j] = optTemp;
		dev_ATE_myOptimalindex[configOffset + j] = myOptimalindex;

		if (dev_ATE_NSsubsets_size[j] > 0)
		{
			for (int i = 0; i < powK; i++){
				dev_ATE_myMinNSVector[(j + configOffset) * powK + i] = dev_ATE_NSsubsets[(NSOffset + myOptimalindex) * powK + i];
			}
		}
	}//end if (j)
*/
}//end FindOPT()

#else

__global__ void FindOPT(int *dev_ATE_elm, int *dev_counterVec, int indexomp, int *dev_roundVec, const int T,
						const int k, const int powK, const int AllTableElemets_size, int *dev_ATE_Csubsets,
						int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, int *dev_zeroVec,
						int *dev_ATE_optVector, int *dev_ATE_optVector_size, int *dev_ATE_myOPT,
						int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, const int i, int *it,
						int *s, int *NS, const int maxSubsetsSize, const int optVectorSize)
{
	//something new and try
	int maxIndex;

	int thread = blockDim.x * blockIdx.x + threadIdx.x;

	int j = thread + indexomp;
	if (thread < dev_counterVec[i]){
//      for(int h=0;h<AllTableElemets[j].NSsubsets.size();h++)   // looking through subset of NSTableElements[j],
																// NSTableElements is the table for all previous OPT. Find all subsets(dependency) of selected job

		int optVecIndex = 0;

		for(int h=0; h < dev_ATE_NSsubsets_size[j]; h++)
		{
			if(gpu_sameVectors(&dev_ATE_NSsubsets[(j * maxSubsetsSize + h) * powK], dev_zeroVec, powK))
			{
				dev_ATE_optVector[j * optVectorSize + optVecIndex] = 0;
				optVecIndex++;
				dev_ATE_optVector_size[j] = optVecIndex;
				break;
			}

			for (int r = 0; r < AllTableElemets_size; r++)
			{
				if (gpu_sameVectors(&dev_ATE_NSsubsets[(j * maxSubsetsSize + h) * powK], &dev_ATE_elm[r * powK], powK))
				{
					//AllTableElemets[j].optVector.push_back(AllTableElemets[r].myOPT);
					dev_ATE_optVector[j * optVectorSize + optVecIndex] = dev_ATE_myOPT[r];
					optVecIndex++;
					dev_ATE_optVector_size[j] = optVecIndex;

					break;
				}
			}
		}

		int minn = 100000;
		int myOptimalindex;
		//for(int pp=0; pp<AllTableElemets[j].optVector.size();pp++)			// find out the OPT from all dependencies.
		for (int pp = 0; pp < dev_ATE_optVector_size[j]; pp++)
		{
			if (dev_ATE_optVector[j * optVectorSize + pp] < minn)
			{
//                    minn=AllTableElemets[j].optVector[pp];
				minn = dev_ATE_optVector[j * optVectorSize + pp];
				myOptimalindex=pp;
			}
		}


		int optTemp=minn+1;
//            AllTableElemets[j].myOPT=optTemp;
		dev_ATE_myOPT[j] = optTemp;
//            AllTableElemets[j].myOptimalindex=myOptimalindex;
		dev_ATE_myOptimalindex[j] = myOptimalindex;

//            if(AllTableElemets[j].NSsubsets.size()>0)
		if (dev_ATE_NSsubsets_size[j] > 0)
		{
//                AllTableElemets[j].myMinNSVector=AllTableElemets[j].NSsubsets[myOptimalindex];
#pragma unroll
			for (int i = 0; i < powK; i++){
				dev_ATE_myMinNSVector[j * powK + i] = dev_ATE_NSsubsets[(j * maxSubsetsSize + myOptimalindex) * powK + i];
			}
			//dev_ATE_myMinNSVector[j] = dev_ATE_NSsubsets[(j * CWhole.SIZE + myOptimalindex) * pow(k,2)];
		}
	}//end if (j)


}//end FindOPT()
#endif

/*
__global__ void LaunchBlocks(int blockOffset, int *dev_ATE_elm, int *dev_counterVec, int *dev_roundVec,
							const int powK, const int cpuId, const int AllTableElemets_size, int *dev_ATE_Csubsets,
							int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, int *dev_zeroVec, int *dev_ATE_myOPT,
							int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, const int blockLvl,
							const int maxSubsetsSize, int *dev_ATE_optVector, int jobsPerBlock,
							int *blockDimSize, const int configOffset, const int T, int *it, int *ss,
							int* NS, int *divisor, int *divisorComp, const int divSize, const int levelsPerBlock,
							const int blockIDInLvl, int *dev_lock1, const int optVectorSize)
*/
__global__ void LaunchBlocks(int blockOffset, int *dev_ATE_elm, int *dev_counterVec, int *dev_roundVec,
							const int powK, const int cpuId, const int AllTableElemets_size, 
							int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, int *dev_zeroVec, int *dev_ATE_myOPT,
							int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, const int blockLvl,
							const int maxSubsetsSize, int jobsPerBlock, int *blockDimSize, const int configOffset, 
							const int T, int *divisor, int *divisorComp, const int divSize, 
							const int levelsPerBlock, const int blockIDInLvl, int *dev_lock1)
{
	int ii = 0;
	int indexomp = 0;
	int *bN = &dev_ATE_elm[configOffset * powK];
	//int *bC = &dev_ATE_Csubsets[configOffset * maxSubsetsSize * powK];
	int *bNS = &dev_ATE_NSsubsets[configOffset * maxSubsetsSize * powK];
	int *bNS_size = &dev_ATE_NSsubsets_size[configOffset];
	//int *bOptVec = &dev_ATE_optVector[configOffset * optVectorSize];
	//int *bOptVec_size = &dev_ATE_optVector_size[configOffset];

	while (ii < levelsPerBlock)		//number of levels = the sum of each block dimension size -1
	{
		int tSize = 64;
		int bSize = 1;

		if (tSize < dev_counterVec[ii]){
			bSize = (tSize + dev_counterVec[ii] - 1) / tSize;
		}
/*
		gpu_generate2<<<bSize, tSize>>>(maxSubsetsSize, bN, powK, bC, bNS, dev_roundVec,
					T, it, ss, NS, bNS_size, dev_counterVec, ii, indexomp, dev_lock1, cpuId,
					blockLvl, blockIDInLvl);
*/					
		gpu_generate2<<<bSize, tSize>>>(maxSubsetsSize, bN, powK, bNS, dev_roundVec, T, 
										bNS_size, dev_counterVec, ii, indexomp, dev_lock1, 
										cpuId, blockLvl, blockIDInLvl);

		cudaDeviceSynchronize();
/*
		FindOPT<<<bSize, tSize>>>(dev_ATE_elm, dev_counterVec, indexomp, dev_roundVec,
								powK, AllTableElemets_size, bC, bNS, bNS_size, dev_zeroVec,
								dev_ATE_myOPT, dev_ATE_myOptimalindex, dev_ATE_myMinNSVector, 
								ii, maxSubsetsSize, blockDimSize, divisor, divisorComp, divSize, 
								jobsPerBlock, cpuId, configOffset, blockLvl, blockIDInLvl,
								bOptVec, optVectorSize);
*/
		FindOPT<<<bSize, tSize>>>(dev_ATE_elm, dev_counterVec, indexomp, dev_roundVec,
								powK, AllTableElemets_size, bNS, bNS_size, dev_zeroVec,
								dev_ATE_myOPT, dev_ATE_myOptimalindex, dev_ATE_myMinNSVector, 
								ii, maxSubsetsSize, blockDimSize, divisor, divisorComp, divSize, 
								jobsPerBlock, cpuId, configOffset, blockLvl, blockIDInLvl);

		indexomp+=dev_counterVec[ii];
		ii++;
		cudaDeviceSynchronize();
	}
}





#ifdef SPLIT
//****************************************************************************************************************
//*********************** This is function for multiple blocks ***************************************************
//****************************************************************************************************************
void gpu_BlockDP(vector<DynamicTable> &AllTableElemets, const int T, const int powK, const int jobsPerBlock,
			const int levelsPerBlock, vector<int> &counterVec, const int LongJobs_size, int *zeroVec,
			int *roundVec, vector<int> &divisor, vector<int> &divisorComp, vector<int> &blockDimSize,
			vector<block> &allBlocks, vector<block> &allBlocksNoZero, vector<int> &blockCounterVec)
{
	cudaSetDevice(0);

	const int thread = omp_get_thread_num();
	const int numThreads = omp_get_num_threads();
	const int batchSize = maxStreamNum/numThreads;

	for (int i=0; i < batchSize; i++)
		cudaStreamCreate(&streams[thread*batchSize+i]);

	int *dev_ATE_elm = 0, *dev_ATE_myOPT = 0, *dev_ATE_myOptimalindex = 0, *dev_ATE_myMinNSVector = 0;
	int *dev_ATE_NSsubsets = 0;
	//int *dev_ATE_Csubsets = 0;
	//int *dev_ATE_optVector = 0;
	int *dev_counterVec = 0;
	int *dev_ATE_NSsubsets_size = 0;
	//int *dev_ATE_optVector_size = 0;
	int *dev_zeroVec = 0, *dev_roundVec = 0;
	//int *it = 0, *ss = 0, *NS = 0;
	int *dev_blockDimSize, *dev_divisor, *dev_divisorComp;
	int *dev_lock1;//, *dev_lock2;
	//int *dev_ifSame;
	
    int ii=0;
    int indexomp=0;
    const int maxSubsetsSize = 64;
    //const int optVectorSize = 64;

/*
	int maxCounterVec = 0;
	int maxIndex = AllTableElemets.size() - 1;
	vector<int> temp;
	for (int p = powK-1; p >= 0; p--)
	{
		if (AllTableElemets[maxIndex].elm[p] != 0)
			temp.push_back(AllTableElemets[maxIndex].elm[p]);
	}

	for (int i = 0; i < temp.size(); i++)
	{
		int a = 1;
		for (int j = 0; j < i + 1; j++)
		{
			a *= temp[j];
		}
//		cout << "Update maxSubsetsSize, current: " << maxSubsetsSize << ", a: " << a <<endl;
		maxSubsetsSize += a;
	}
*/
	int maxBlockLvlSize = 0;
	int maxInBlockLvlSize = 0;

	for (int i = 0; i < blockCounterVec.size(); i++)
	{
		if (blockCounterVec[i] > maxBlockLvlSize)
			maxBlockLvlSize = blockCounterVec[i];
	}

	for (int i = 0; i < counterVec.size(); i++)
	{
		if (counterVec[i] > maxInBlockLvlSize)
			maxInBlockLvlSize = counterVec[i];
	}

#ifdef _DEVICE_DEBUG
	cout << "thread: " << thread << ", maxBlockLvlSize: " << maxBlockLvlSize << endl;
	cout << "thread: " << thread << ", maxInBlockLvlSize: " << maxInBlockLvlSize << endl;
	cout << "thread: " << thread << ", AllTableSize: " << AllTableElemets.size() << ", maxSubsetsSize: " << maxSubsetsSize << endl;
#endif


	//arrays on device
	//gpuErrchk(cudaMalloc((void**)&dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_elm, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myMinNSVector, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myOPT, AllTableElemets.size() * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int)));
	//gpuErrchk(cudaMalloc((void**)&dev_ATE_optVector, AllTableElemets.size() * optVectorSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_counterVec, counterVec.size() * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_zeroVec, (powK) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_roundVec, (powK) * sizeof(int)));
	//gpuErrchk(cudaMalloc((void**)&it, batchSize * maxBlockLvlSize * maxInBlockLvlSize * powK * sizeof(int)));
	//gpuErrchk(cudaMalloc((void**)&ss, batchSize * maxBlockLvlSize * maxInBlockLvlSize * powK * sizeof(int)));
	//gpuErrchk(cudaMalloc((void**)&NS, batchSize * maxBlockLvlSize * maxInBlockLvlSize * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_NSsubsets_size, AllTableElemets.size() * sizeof(int)));
    //gpuErrchk(cudaMalloc((void**)&dev_ATE_optVector_size, AllTableElemets.size() * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_blockDimSize, blockDimSize.size() * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_divisor, divisor.size() * sizeof(int)))
    gpuErrchk(cudaMalloc((void**)&dev_divisorComp, powK * sizeof(int)))
    gpuErrchk(cudaMalloc((void**)&dev_lock1, AllTableElemets.size() * sizeof(int)));
    //gpuErrchk(cudaMalloc((void**)&dev_lock2, AllTableElemets.size() * sizeof(int)));
    //gpuErrchk(cudaMalloc((void**)&dev_ifSame, AllTableElemets.size() * sizeof(int)));
	//cout << "thread: " << omp_get_thread_num() << ", cuda initial is done." << " Start memcpy! AllTableElemets.size: " << AllTableElemets.size() << endl;


	int *ATE_myOPT = new int[AllTableElemets.size()];

	for (int i = 0; i < AllTableElemets.size(); i++){
		//gpuErrchk(cudaMemcpyAsync(&dev_ATE_elm[i * powK], &AllTableElemets[i].elm[0], powK * sizeof(int), cudaMemcpyHostToDevice, streams[thread*batchSize]));
		gpuErrchk(cudaMemcpy(&dev_ATE_elm[i * powK], &AllTableElemets[i].elm[0], powK * sizeof(int), cudaMemcpyHostToDevice));
		ATE_myOPT[i] = AllTableElemets[i].myOPT;
	}


	//gpuErrchk(cudaMemcpyAsync(dev_zeroVec, zeroVec, powK*sizeof(int), cudaMemcpyHostToDevice));
	
	gpuErrchk(cudaMemsetAsync(dev_zeroVec, 0, powK*sizeof(int), streams[thread*batchSize]));
	gpuErrchk(cudaMemsetAsync(dev_lock1, 0, AllTableElemets.size() * sizeof(int), streams[thread*batchSize]));
	//gpuErrchk(cudaMemsetAsync(dev_lock2, 0, AllTableElemets.size() * sizeof(int), streams[thread*batchSize]));
	gpuErrchk(cudaMemsetAsync(dev_ATE_NSsubsets_size, 0, AllTableElemets.size()*sizeof(int), streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(dev_roundVec, roundVec, powK*sizeof(int), cudaMemcpyHostToDevice, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(dev_counterVec, &counterVec[0], counterVec.size() * sizeof(int), cudaMemcpyHostToDevice, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(dev_ATE_myOPT, ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyHostToDevice, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(dev_blockDimSize, &blockDimSize[0], blockDimSize.size()*sizeof(int), cudaMemcpyHostToDevice, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(dev_divisor, &divisor[0], divisor.size() * sizeof(int), cudaMemcpyHostToDevice, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(dev_divisorComp, &divisorComp[0], powK * sizeof(int), cudaMemcpyHostToDevice, streams[thread*batchSize]));
/*
	gpuErrchk(cudaMemset(dev_zeroVec, 0, powK*sizeof(int)));
	gpuErrchk(cudaMemcpy(dev_roundVec, roundVec, powK*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_counterVec, &counterVec[0], counterVec.size() * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_ATE_myOPT, ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_blockDimSize, &blockDimSize[0], blockDimSize.size()*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_divisor, &divisor[0], divisor.size() * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_divisorComp, &divisorComp[0], powK * sizeof(int), cudaMemcpyHostToDevice));
*/	
	int b = 0;
	while (ii < blockCounterVec.size())
	{
		for (int tt = 0; tt < blockCounterVec[ii]; tt++)
		{
			//each stream process only one block at a time. Dispatch blocks within the same level to different streams.
			int subStream = tt % batchSize;
			//streamOffset is the global memory offset for one cpu thread, it returns the memory offset for the current stream.
			//int streamOffset = subStream * maxBlockLvlSize * maxInBlockLvlSize * powK;
			//configLvlOffset returns the memory offset for the block tt within the same stream.
			//Since each stream is allocated to the MAX memory space, only some memory are used. For example maxBlockLvlSize = 100, but we
			//have only 45 blocks in the current level. Thus, configLvlOffset helps find the exact memory offset.
			//int configLvlOffset = tt * maxInBlockLvlSize * powK;
			//This offset is used by arrays it, ss, and NS as temporary storage.
			//int totalOffset = streamOffset + configLvlOffset;

			if (allBlocks[b].mySUM == ii)
			{
#ifdef _DEVICE_DEBUG
				if (thread == 0){
					cout << "thread: " << thread << ", allBlocks[" << b << "]: ";
					for (int i=0; i<powK; i++)
					{
						cout << allBlocks[b].elm[i] << " ";
					}
					cout << "mySum: " << allBlocks[b].mySUM << endl;
					cout << "thread: " << thread << ", blockNoZero[" << b << "]: ";
					for (int i=0; i<allBlocksNoZero[b].elm.size(); i++)
					{
						cout << allBlocksNoZero[b].elm[i] << " ";
					}
					cout << endl;
				}
#endif
				//blockOffset returns the current block ID.
				int blockOffset = gpu_blockOffsetNoZero(&allBlocksNoZero[b].elm[0], &divisor[0],
												allBlocksNoZero[b].elm.size(), divisor.size());
				//configOffset returns the number of jobs before the current block. This is not memory offset.
				int configOffset = blockOffset * jobsPerBlock;

				//cout << "cpu: " << thread << ", blockOffset: " << blockOffset << ", configOffset: " << configOffset << endl;

				//allBlocks contains the block ID which is needed to calculate the beginning vector position in AllTableElemets.
/*				LaunchBlocks<<<1, 1, 0, streams[thread*batchSize+subStream]>>>(blockOffset,
						dev_ATE_elm, dev_counterVec, dev_roundVec, powK, thread,
						AllTableElemets.size(), dev_ATE_Csubsets, dev_ATE_NSsubsets,
						dev_ATE_NSsubsets_size, dev_zeroVec, dev_ATE_myOPT, dev_ATE_myOptimalindex,
						dev_ATE_myMinNSVector, ii, maxSubsetsSize, dev_ATE_optVector, jobsPerBlock,
						dev_blockDimSize, configOffset, T, &it[totalOffset], &ss[totalOffset],
						&NS[totalOffset], dev_divisor, dev_divisorComp, divisor.size(), 
						counterVec.size(), tt, dev_lock1, optVectorSize);    */
				LaunchBlocks<<<1, 1, 0, streams[thread*batchSize+subStream]>>>(blockOffset,
						dev_ATE_elm, dev_counterVec, dev_roundVec, powK, thread,
						AllTableElemets.size(), dev_ATE_NSsubsets, dev_ATE_NSsubsets_size, 
						dev_zeroVec, dev_ATE_myOPT, dev_ATE_myOptimalindex, dev_ATE_myMinNSVector, 
						ii, maxSubsetsSize, jobsPerBlock, dev_blockDimSize, configOffset, T, 
						dev_divisor, dev_divisorComp, divisor.size(), counterVec.size(), tt, dev_lock1);
				//break;
				b++;
			}

		}

		//for (int j = 0; j < blockCounterVec[ii]; j++)
		for (int subStream = 0; subStream < batchSize; subStream++)
		{
			//int subStream = j % batchSize;
			cudaStreamSynchronize(streams[thread*batchSize+subStream]);
		}

		indexomp += blockCounterVec[ii];
		ii++;
	}

/*********************  GPU code to update AllTableElement  ********************************/
	int *temp_NSsubsets = new int[AllTableElemets.size() * maxSubsetsSize * powK];
	//int *temp_Csubsets = new int[AllTableElemets.size() * maxSubsetsSize * powK];
	int *temp_myOPT = new int[AllTableElemets.size()];
	int *temp_myOptIndex = new int[AllTableElemets.size()];
	int *temp_myMinNSVector = new int[AllTableElemets.size() * powK];
	//int *temp_optVector = new int[AllTableElemets.size() * optVectorSize];

//	cout << "AllTableSize: " << AllTableElemets.size() << ", maxSubsetsSize: " << maxSubsetsSize << ", powK: " << powK << endl;

	gpuErrchk(cudaMemcpyAsync(temp_NSsubsets, dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	//gpuErrchk(cudaMemcpyAsync(temp_Csubsets, dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_myOPT, dev_ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_myOptIndex, dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_myMinNSVector, dev_ATE_myMinNSVector, AllTableElemets.size() * powK * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	//gpuErrchk(cudaMemcpyAsync(temp_optVector, dev_ATE_optVector, AllTableElemets.size() * optVectorSize * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
/*
	gpuErrchk(cudaMemcpy(temp_NSsubsets, dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_Csubsets, dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_myOPT, dev_ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_myOptIndex, dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_myMinNSVector, dev_ATE_myMinNSVector, AllTableElemets.size() * powK * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_optVector, dev_ATE_optVector, AllTableElemets.size() * optVectorSize * sizeof(int), cudaMemcpyDeviceToHost));
*/
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	for (int i = 0; i < AllTableElemets.size(); i++)
	{
		AllTableElemets[i].myOPT = temp_myOPT[i];
		AllTableElemets[i].myOptimalindex = temp_myOptIndex[i];
		int begin = 0, end = maxSubsetsSize * powK;
		while (begin != end)
		{
			AllTableElemets[i].NSsubsets.push_back(std::vector<int>(&temp_NSsubsets[(i * maxSubsetsSize) * powK], &temp_NSsubsets[(i * maxSubsetsSize + 1) * powK]));
			//AllTableElemets[i].Csubsets.push_back(std::vector<int>(&temp_Csubsets[(i * maxSubsetsSize) * powK], &temp_Csubsets[(i * maxSubsetsSize + 1) * powK]));
			begin += powK;
		}
		//AllTableElemets[i].optVector.insert(AllTableElemets[i].optVector.end(), &temp_optVector[i * optVectorSize], &temp_optVector[(i + 1) * optVectorSize]);
		AllTableElemets[i].myMinNSVector.insert(AllTableElemets[i].myMinNSVector.end(), &temp_myMinNSVector[i * powK], &temp_myMinNSVector[(i + 1) * powK]);
	}

	gettimeofday(&t2, NULL);
	//cout << "memory transfer to vectors Runtime: "  << t2.tv_sec - t1.tv_sec << endl;


	delete[] ATE_myOPT;
	delete[] temp_NSsubsets;
	//delete[] temp_Csubsets;
	delete[] temp_myOPT;
	delete[] temp_myOptIndex;
	delete[] temp_myMinNSVector;
	//delete[] temp_optVector;
	//cudaFree(dev_ATE_Csubsets);
	cudaFree(dev_ATE_NSsubsets);
	cudaFree(dev_ATE_elm);
	cudaFree(dev_ATE_myMinNSVector);
	cudaFree(dev_ATE_myOPT);
	cudaFree(dev_ATE_myOptimalindex);
	//cudaFree(dev_ATE_optVector);
	cudaFree(dev_counterVec);
	cudaFree(dev_zeroVec);
	cudaFree(dev_roundVec);
	//cudaFree(it);
	//cudaFree(ss);
	//cudaFree(NS);
	cudaFree(dev_ATE_NSsubsets_size);
	//cudaFree(dev_ATE_optVector_size);
	cudaFree(dev_blockDimSize);
	cudaFree(dev_divisor);
	cudaFree(dev_lock1);
	//cudaFree(dev_lock2);

	for (int i=0; i<batchSize; i++)
		cudaStreamDestroy(streams[thread*batchSize+i]);
}


#else
void gpu_DP(vector<DynamicTable> &AllTableElemets, const int T, const int k, const int powK, const int maxSumValue,
			vector<int> &counterVec, const int LongJobs_size, int *zeroVec, int *roundVec)
{
	cudaSetDevice(0);

	const int thread = omp_get_thread_num();
	const int numThreads = omp_get_num_threads();
	const int batchSize = maxStreamNum/numThreads;

	for (int i=0; i < batchSize; i++)
		cudaStreamCreate(&streams[thread*batchSize+i]);

	int *dev_ATE_elm = 0, *dev_ATE_myOPT = 0, *dev_ATE_myOptimalindex = 0, *dev_ATE_myMinNSVector = 0;
	int *dev_ATE_NSsubsets = 0, *dev_ATE_Csubsets = 0;
	int *dev_ATE_optVector = 0;
	int *dev_counterVec = 0;
	int *dev_ATE_NSsubsets_size = 0;
	int *dev_ATE_optVector_size = 0;
	int *dev_zeroVec = 0, *dev_roundVec = 0;
	int *it = 0, *ss = 0, *NS = 0;

    int ii=0;
    int indexomp=0;
    int maxSubsetsSize = 0;
	const int optVectorSize = 64;


//	InitGPUData(powK, LongJobs_size, AllTableElemets, zeroVec, roundVec, &counterVec[0], maxSubsetsSize, optVectorSize, counterVec.size(),
//				&dev_ATE_elm, &dev_ATE_myOPT, &dev_ATE_myOptimalindex, &dev_ATE_myMinNSVector, &dev_ATE_NSsubsets, &dev_ATE_Csubsets,
//				&dev_ATE_optVector, &dev_counterVec, &dev_ATE_NSsubsets_size, &dev_ATE_optVector_size, &dev_zeroVec, &dev_roundVec, &it, &ss, &NS);

	int maxIndex = AllTableElemets.size() - 1;
	int maxCounterVec = 0;
	vector<int> temp;
//	for (vector<int>::const_iterator pt = AllTableElemets[maxIndex].elm.end(); pt != AllTableElemets[maxIndex].elm.begin(); --pt)
	for (int p = powK-1; p >= 0; p--)
	{
		if (AllTableElemets[maxIndex].elm[p] != 0)
			temp.push_back(AllTableElemets[maxIndex].elm[p]);
	}

	for (int i = 0; i < temp.size(); i++)
	{
		int a = 1;
		for (int j = 0; j < i + 1; j++)
		{
			a *= temp[j];
		}
//		cout << "Update maxSubsetsSize, current: " << maxSubsetsSize << ", a: " << a <<endl;
		maxSubsetsSize += a;
	}

	maxSubsetsSize = 128;


	for (int i = 0; i < counterVec.size(); i++)
	{
		if (counterVec[i] > maxCounterVec)
			maxCounterVec = counterVec[i];
	}

//	cout << "AllTableSize: " << AllTableElemets.size() << ", maxSubsetsSize: " << maxSubsetsSize << ", powK: " << powK << ", tempSize: " << temp.size() << endl;

	//arrays on device
	gpuErrchk(cudaMalloc((void**)&dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_elm, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myMinNSVector, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myOPT, AllTableElemets.size() * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_optVector, AllTableElemets.size() * optVectorSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_counterVec, counterVec.size() * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_zeroVec, (powK) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_roundVec, (powK) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&it, batchSize * maxCounterVec * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&ss, batchSize * maxCounterVec * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&NS, batchSize * maxCounterVec * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_NSsubsets_size, AllTableElemets.size() * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_ATE_optVector_size, AllTableElemets.size() * sizeof(int)));


    //*********** For gpu_sameVector and gpu_genSubConfigs ****************************************************
    int *ifsame;
    int *dev_subConfigSize;
    int *dev_countC, *dev_countNS;
    gpuErrchk(cudaMalloc((void**)&ifsame, batchSize * maxCounterVec * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_subConfigSize, batchSize * maxCounterVec * powK * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_countC, AllTableElemets.size() * maxSubsetsSize * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_countNS, AllTableElemets.size() * maxSubsetsSize * sizeof(int)));
	gpuErrchk(cudaMemsetAsync(dev_countC, 0, AllTableElemets.size() * maxSubsetsSize *sizeof(int), streams[thread*batchSize]));
	gpuErrchk(cudaMemsetAsync(dev_countNS, 0, AllTableElemets.size() * maxSubsetsSize *sizeof(int), streams[thread*batchSize]));
	//*********************************************************************************************************


	//cout << "thread: " << omp_get_thread_num() << ", cuda initial is done." << " Start memcpy! AllTableElemets.size: " << AllTableElemets.size() << endl;


	int *ATE_myOPT = new int[AllTableElemets.size()];

	for (int i = 0; i < AllTableElemets.size(); i++){
		gpuErrchk(cudaMemcpyAsync(&dev_ATE_elm[i * powK], &AllTableElemets[i].elm[0], powK * sizeof(int), cudaMemcpyHostToDevice, streams[thread*batchSize]));
		ATE_myOPT[i] = AllTableElemets[i].myOPT;
	}

	//gpuErrchk(cudaMemcpyAsync(dev_zeroVec, zeroVec, powK*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemsetAsync(dev_zeroVec, 0, powK*sizeof(int), streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(dev_roundVec, roundVec, powK*sizeof(int), cudaMemcpyHostToDevice, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(dev_counterVec, &counterVec[0], counterVec.size() * sizeof(int), cudaMemcpyHostToDevice, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(dev_ATE_myOPT, ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyHostToDevice, streams[thread*batchSize]));


//	cout << ", LongJob size: " << LongJobs_size << ", maxSumValue: " << maxSumValue << endl;

	while (ii < maxSumValue+1)		//number of levels = number of jobs + 1
    {
		int tSize = 256;
		int bSize = 1;

		for (int ptr=0; ptr < batchSize; ptr++)
		{
			if (ii < maxSumValue+1)
			{
				if (tSize < counterVec[ii]){
					bSize = (tSize + counterVec[ii] - 1) / tSize;
				}

				int sizeOffset = ptr * maxCounterVec * powK;
//		std::cout << "counterVec[" << ii << "]: " << counterVec[ii] << ", indexomp: " << indexomp << std::endl;
//				gpu_generate2<<<bSize, tSize, 0, streams[thread*batchSize+ptr]>>>(maxSubsetsSize,
//						 dev_ATE_elm, powK, dev_ATE_Csubsets, dev_ATE_NSsubsets, dev_roundVec, T,
//						 dev_ATE_NSsubsets_size, dev_counterVec, ii, indexomp,
//						 dev_subConfigSize, dev_countC, dev_countNS);

				gpu_generate2<<<bSize, tSize, 0, streams[thread*batchSize+ptr]>>>(maxSubsetsSize,
							dev_ATE_elm, powK, dev_ATE_Csubsets, dev_ATE_NSsubsets, dev_roundVec,
							T, it, ss, NS, dev_ATE_NSsubsets_size, dev_counterVec, ii, indexomp);

				FindOPT<<<bSize, tSize, 0, streams[thread*batchSize+ptr]>>>(dev_ATE_elm,
													dev_counterVec, indexomp, dev_roundVec, powK,
													AllTableElemets.size(), dev_ATE_Csubsets, dev_ATE_NSsubsets,
													dev_ATE_NSsubsets_size, dev_zeroVec, dev_ATE_optVector,
													dev_ATE_optVector_size, dev_ATE_myOPT, dev_ATE_myOptimalindex,
													dev_ATE_myMinNSVector, ii, maxSubsetsSize, optVectorSize,
													dev_countC, dev_countNS);

//				FindOPT<<<bSize, tSize, 0, streams[thread*batchSize+ptr]>>>(dev_ATE_elm,
//									dev_counterVec, indexomp, dev_roundVec, T, k, powK,
//									AllTableElemets.size(), dev_ATE_Csubsets, dev_ATE_NSsubsets,
//									dev_ATE_NSsubsets_size, dev_zeroVec, dev_ATE_optVector,
//									dev_ATE_optVector_size, dev_ATE_myOPT, dev_ATE_myOptimalindex,
//									dev_ATE_myMinNSVector, ii, &it[sizeOffset], &ss[sizeOffset],
//									&NS[sizeOffset], maxSubsetsSize, optVectorSize);

				cudaStreamSynchronize(streams[thread*batchSize+ptr]);

				indexomp+=counterVec[ii];
				ii++;
			}
		}
    }

/*********************  GPU code to update AllTableElement  ********************************/
	int *temp_NSsubsets, *temp_Csubsets, *temp_myOPT, *temp_myOptIndex, *temp_myMinNSVector, *temp_optVector;
	//int *temp_countC, *temp_countNS;
	temp_NSsubsets = new int[AllTableElemets.size() * maxSubsetsSize * powK];
	temp_Csubsets = new int[AllTableElemets.size() * maxSubsetsSize * powK];
	temp_myOPT = new int[AllTableElemets.size()];
	temp_myOptIndex = new int[AllTableElemets.size()];
	temp_myMinNSVector = new int[AllTableElemets.size() * powK];
	temp_optVector = new int[AllTableElemets.size() * optVectorSize];
	//temp_countC = new int[AllTableElemets.size() * maxSubsetsSize];
	//temp_countNS = new int[AllTableElemets.size() * maxSubsetsSize];

//	cout << "AllTableSize: " << AllTableElemets.size() << ", maxSubsetsSize: " << maxSubsetsSize << ", powK: " << powK << endl;

	gpuErrchk(cudaMemcpyAsync(temp_NSsubsets, dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_Csubsets, dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_myOPT, dev_ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_myOptIndex, dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_myMinNSVector, dev_ATE_myMinNSVector, AllTableElemets.size() * powK * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_optVector, dev_ATE_optVector, AllTableElemets.size() * optVectorSize * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	//gpuErrchk(cudaMemcpyAsync(temp_countC, dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	//gpuErrchk(cudaMemcpyAsync(temp_countNS, dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	for (int i = 0; i < AllTableElemets.size(); i++)
	{
		AllTableElemets[i].myOPT = temp_myOPT[i];
		AllTableElemets[i].myOptimalindex = temp_myOptIndex[i];
		int begin = 0, end = maxSubsetsSize * powK;
		while (begin != end)
		{
			AllTableElemets[i].NSsubsets.push_back(std::vector<int>(&temp_NSsubsets[(i * maxSubsetsSize) * powK], &temp_NSsubsets[(i * maxSubsetsSize + 1) * powK]));
			AllTableElemets[i].Csubsets.push_back(std::vector<int>(&temp_Csubsets[(i * maxSubsetsSize) * powK], &temp_Csubsets[(i * maxSubsetsSize + 1) * powK]));
			begin += powK;
		}
		AllTableElemets[i].optVector.insert(AllTableElemets[i].optVector.end(), &temp_optVector[i * optVectorSize], &temp_optVector[(i + 1) * optVectorSize]);
		AllTableElemets[i].myMinNSVector.insert(AllTableElemets[i].myMinNSVector.end(), &temp_myMinNSVector[i * powK], &temp_myMinNSVector[(i + 1) * powK]);
	}

	gettimeofday(&t2, NULL);
	cout << "memory transfer to vectors Runtime: "  << t2.tv_sec - t1.tv_sec << endl;


	delete[] ATE_myOPT;
	delete[] temp_NSsubsets;
	delete[] temp_Csubsets;
	delete[] temp_myOPT;
	delete[] temp_myOptIndex;
	delete[] temp_myMinNSVector;
	delete[] temp_optVector;
	//delete[] temp_countC;
	//delete[] temp_countNS;
	cudaFree(dev_ATE_Csubsets);
	cudaFree(dev_ATE_NSsubsets);
	cudaFree(dev_ATE_elm);
	cudaFree(dev_ATE_myMinNSVector);
	cudaFree(dev_ATE_myOPT);
	cudaFree(dev_ATE_myOptimalindex);
	cudaFree(dev_ATE_optVector);
	cudaFree(dev_counterVec);
	cudaFree(dev_zeroVec);
	cudaFree(dev_roundVec);
	cudaFree(it);
	cudaFree(ss);
	cudaFree(NS);
	cudaFree(dev_ATE_NSsubsets_size);
	cudaFree(dev_ATE_optVector_size);

	cudaFree(ifsame);
	cudaFree(dev_subConfigSize);
	cudaFree(dev_countC);
	cudaFree(dev_countNS);

	for (int i=0; i<batchSize; i++)
		cudaStreamDestroy(streams[thread*batchSize+i]);
}
#endif
