#include "DPCUDA.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


const int maxStreamNum = 32;
cudaStream_t streams[maxStreamNum];

	
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

	
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

__device__ void gpu_generate2(int *Ntemp, const int Ntemp_size, int *Ctemp, int *NMinusStemp, int *dev_roundVec, 
							  const int T, const int powK, int *it, int *s, int *NS, int *subsets_size, const int thread){
	//vector<int> it(Ntemp.size(), 0);
	//int it[Ntemp_size];
	int counterNS = 0, counterC = 0;
#pragma unroll
	for (int i = 0; i < Ntemp_size; i++)
	{
		it[i] = 0;
	}

    do {
        //int s[Ntemp_size];
#pragma unroll
        for (int i = 0; i < Ntemp_size; i++)
        {
			s[i] = it[i];
		}
        //for(vector<int>::const_iterator i = it.begin(); i != it.end(); ++i)
        //{
        //    s.push_back(*i);
        //}
        //Cwhole.push_back(s);
        int sSum=gpu_sumFun(s, dev_roundVec, powK);
			
        if(sSum <= T)
        {
#pragma unroll
			for (int i = 0; i < Ntemp_size; i++)
			{
				Ctemp[counterC*Ntemp_size + i] = s[i];
			}
			counterC++;
            //Ctemp.push_back(s);
            
            //int NS[Ntemp_size];
#pragma unroll
            for(int j=0; j<powK; j++)
            {
                NS[j] = Ntemp[j]-s[j];
            }
           
            
            if(gpu_sameVectors(NS, Ntemp, Ntemp_size)){
                continue;
			}
#pragma unroll			
			for (int i = 0; i < Ntemp_size; i++)
			{
				NMinusStemp[counterNS * Ntemp_size + i] = NS[i];
			}
            //NMinusStemp.push_back(NS);
			
			counterNS++;
        }
    }while (gpu_increase(Ntemp, it, Ntemp_size));
    
    *subsets_size = counterNS;
}

//for(int j=indexomp;j< (counterVec[i] + indexomp) ;j++)			// this is to determine the job for each level
		
__global__ void FindOPT(int *dev_ATE_elm, int *dev_counterVec, int indexomp, int *dev_roundVec, const int T, 
						const int k, const int powK, const int AllTableElemets_size, int *dev_ATE_Csubsets, 
						int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, int *dev_zeroVec, 
						int *dev_ATE_optVector, int *dev_ATE_optVector_size, int *dev_ATE_myOPT, 
						int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, const int i, int *it, 
						int *s, int *NS, const int maxSubsetsSize, const int optVectorSize)
{		
		int thread = blockDim.x * blockIdx.x + threadIdx.x;
				
		int j = thread + indexomp;
		if (thread < dev_counterVec[i]){
            //vector<vector<int> > Ctemp;
            //vector<vector<int> > NMinusStemp;
            //vector<vector<int> > Cwhole;
            //generate2(AllTableElemets[j].elm,Ctemp,NMinusStemp);
			
            gpu_generate2(&dev_ATE_elm[j * powK], powK, &dev_ATE_Csubsets[j * maxSubsetsSize * powK],
            			&dev_ATE_NSsubsets[j * maxSubsetsSize * powK], dev_roundVec, T, powK,
            			&it[thread * powK], &s[thread * powK], &NS[thread * powK],
            			&dev_ATE_NSsubsets_size[j], thread);     //dev_ATE_elm_size[j] = Ntemp.size() = powK
			
			__syncthreads();
			

//            AllTableElemets[j].NSsubsets=NMinusStemp; 	//ni-si
//            AllTableElemets[j].Csubsets=Ctemp;		//configurations
            
//            for(int h=0;h<AllTableElemets[j].NSsubsets.size();h++)   // looking through subset of NSTableElements[j], 
																	// NSTableElements is the table for all previous OPT. Find all subsets(dependency) of selected job
			int optVecIndex = 0;
			for(int h=0; h < dev_ATE_NSsubsets_size[j]; h++)
            {
//                if(AllTableElemets[j].NSsubsets[h]==zeroVec)   // if subset is zero Vector , its OPT is 0
				if(gpu_sameVectors(&dev_ATE_NSsubsets[(j * maxSubsetsSize + h) * powK], dev_zeroVec, powK))
                {
                    //AllTableElemets[j].optVector.push_back(0);
                    dev_ATE_optVector[j * optVectorSize + optVecIndex] = 0;
                    optVecIndex++;
                    dev_ATE_optVector_size[j] = optVecIndex;                                  
                    break;
                }             
                
                //if(AllTableElemets[j].NSsubsets[h]==AllTableElemets[j].elm)   // if NSsubsets[h] is equal to NSTableElements[j] (itself) 
																			//( the one that we are doing operation for it ) ----> break (not interested )
																			// check if it is itself, if yes, ignore OPT of job itself.
				if(gpu_sameVectors(&dev_ATE_NSsubsets[(j * maxSubsetsSize + h) * powK], &dev_ATE_elm[j * powK], powK) ){
                    dev_ATE_optVector_size[j] = optVecIndex;

                    break;
				}
								
                //for(int r=0; r<AllTableElemets.size();r++)        // to find the match in the NSTableElements for reaching OPT
																//dependencies may not be consectively stored in the table, so have to go through the whole
																//table (AllTableElemets) and find them (matched to AllTableElemets[j].NSsubsets[h]).
				for (int r = 0; r < AllTableElemets_size; r++)
                {					
                    //if(AllTableElemets[j].NSsubsets[h]==AllTableElemets[r].elm)   // if found match of NSsubsets[h], copy its OPT and break 

                   
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
	
	maxSubsetsSize = 256;


	for (int i = 0; i < counterVec.size(); i++)
	{
		cout <<"thread: " << omp_get_thread_num() << ", counterVec[" << i << "]: " << counterVec[i] << endl;
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
				FindOPT<<<bSize, tSize, 0, streams[thread*batchSize+ptr]>>>(dev_ATE_elm,
									dev_counterVec, indexomp, dev_roundVec, T, k, powK,
									AllTableElemets.size(), dev_ATE_Csubsets, dev_ATE_NSsubsets,
									dev_ATE_NSsubsets_size, dev_zeroVec, dev_ATE_optVector,
									dev_ATE_optVector_size, dev_ATE_myOPT, dev_ATE_myOptimalindex,
									dev_ATE_myMinNSVector, ii, &it[sizeOffset], &ss[sizeOffset],
									&NS[sizeOffset], maxSubsetsSize, optVectorSize);

				cudaStreamSynchronize(streams[thread*batchSize+ptr]);

				indexomp+=counterVec[ii];
				ii++;
			}
		}
    } 
    
//	for (int i=0; i<batchSize; i++)
//	{
//		cudaStreamSynchronize(streams[thread*batchSize+i]);
//	}

/*********************  GPU code to update AllTableElement  ********************************/
	int *temp_NSsubsets, *temp_Csubsets, *temp_myOPT, *temp_myOptIndex, *temp_myMinNSVector, *temp_optVector;
	temp_NSsubsets = new int[AllTableElemets.size() * maxSubsetsSize * powK];
	temp_Csubsets = new int[AllTableElemets.size() * maxSubsetsSize * powK];
	temp_myOPT = new int[AllTableElemets.size()];
	temp_myOptIndex = new int[AllTableElemets.size()];
	temp_myMinNSVector = new int[AllTableElemets.size() * powK];
	temp_optVector = new int[AllTableElemets.size() * optVectorSize];
	
//	cout << "AllTableSize: " << AllTableElemets.size() << ", maxSubsetsSize: " << maxSubsetsSize << ", powK: " << powK << endl;
	
	gpuErrchk(cudaMemcpyAsync(temp_NSsubsets, dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_Csubsets, dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_myOPT, dev_ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_myOptIndex, dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_myMinNSVector, dev_ATE_myMinNSVector, AllTableElemets.size() * powK * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	gpuErrchk(cudaMemcpyAsync(temp_optVector, dev_ATE_optVector, AllTableElemets.size() * optVectorSize * sizeof(int), cudaMemcpyDeviceToHost, streams[thread*batchSize]));
	
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

	for (int i=0; i<batchSize; i++)
		cudaStreamDestroy(streams[thread*batchSize+i]);
}
