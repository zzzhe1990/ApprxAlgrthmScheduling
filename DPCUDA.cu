#include "DPCUDA.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

int *dev_ATE_elm, *dev_ATE_myOPT, *dev_ATE_myOptimalindex, *dev_ATE_myMinNSVector;
	int *dev_ATE_NSsubsets, *dev_ATE_Csubsets;
	int *dev_ATE_optVector;
	int *dev_counterVec;
	int *dev_ATE_NSsubsets_size;
	int *dev_ATE_optVector_size;
	int *dev_zeroVec, *dev_roundVec;
	int *it, *ss, *NS;
	
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
	
void InitGPUData(int powK, int LongJobs_size, vector<DynamicTable> &AllTableElemets, int *zeroVec, 
				 int *roundVec, int *counterVec, int &maxSubsetsSize, const int maxSumValue, const int counterVecSize)
{
	cout << "At the beginning of InitGPUData. maxSubsetsSize: " << maxSubsetsSize << endl;
	
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
	gpuErrchk(cudaMalloc((void**)&dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_elm, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myMinNSVector, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myOPT, AllTableElemets.size() * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_optVector, AllTableElemets.size() * (maxSumValue +1) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_counterVec, (LongJobs_size + 1) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_zeroVec, (powK) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_roundVec, (powK) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&it, maxCounterVec * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&ss, maxCounterVec * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&NS, maxCounterVec * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_NSsubsets_size, AllTableElemets.size() * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_ATE_optVector_size, AllTableElemets.size() * sizeof(int)));
    
	int *ATE_myOPT = new int[AllTableElemets.size()];
	for (int i = 0; i < AllTableElemets.size(); i++){
		gpuErrchk(cudaMemcpy(&dev_ATE_elm[i * powK], &AllTableElemets[i].elm[0], powK * sizeof(int), cudaMemcpyHostToDevice));
		ATE_myOPT[i] = AllTableElemets[i].myOPT;
	}
	
	gpuErrchk(cudaMemcpy(dev_zeroVec, zeroVec, powK*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_roundVec, roundVec, powK*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_counterVec, counterVec, (LongJobs_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_ATE_myOPT, ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyHostToDevice));
	
	delete(ATE_myOPT);
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

__device__ int gpu_increase(const int *Ntemp, int *it, int Ntemp_size, int thread, int counter)
{
	
	printf("At the beginning of gpu_increase, thread: %d, counter: %d\n", thread, counter);
	
	for (int i = 0, size = Ntemp_size; i != size; ++i) 
	{
		const int index = size - 1 - i;
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
#pragma unroll
	for(int i=0; i<powK; i++)
	{
		summ= summ + A[i]*B[i];
	}
	return summ;
}

__device__ void gpu_generate2(int *Ntemp, const int Ntemp_size, int *Ctemp, int *NMinusStemp, int *dev_roundVec, 
							  const int T, const int powK, int *it, int *s, int *NS, int *subsets_size, const int thread){
	
	//Ntemp_size = pow(k,2)
	
	//vector<int> it(Ntemp.size(), 0);
	//int it[Ntemp_size];
	int counter = 0;
#pragma unroll
	for (int i = 0; i < Ntemp_size; i++)
	{
		it[i] = 0;
	}
		
//	if (thread == 0)
	{
		printf("In gpu_generate2 before do while loop, thread: %d.\n", thread);
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
		
		
//        if(thread == 0)
        {
			printf("In gpu_generate2 in do while loop, counter: %d, thread: %d\n", counter, thread);
		}
			
        if(sSum <= T)
        {
			for (int i = 0; i < Ntemp_size; i++)
			{
				Ctemp[counter*Ntemp_size + i] = s[i];
			}
            //Ctemp.push_back(s);
            
//            if(thread == 0)
            {
				printf("Ctemp is updated successfully, counter: %d, thread: %d\n", counter, thread);
            }
            
            //int NS[Ntemp_size];
            for(int j=0; j<powK; j++)
            {
                NS[j] = Ntemp[j]-s[j];
            }
            
//            if(thread == 0)
            {
				printf("NS is updated successfully, counter: %d, thread: %d\n", counter, thread);
            }
            
            if(gpu_sameVectors(NS, Ntemp, Ntemp_size)){
				
//				if(thread == 0)
				{
					printf("sameVector returns true, counter: %d, thread: %d\n", counter, thread);
				}
				
                continue;
			}
			
			
//			if(thread == 0)
			{
				printf("sameVector returns false, counter: %d, thread: %d\n", counter, thread);
			}
			
			for (int i = 0; i < Ntemp_size; i++)
			{
				NMinusStemp[counter * Ntemp_size + i] = NS[i];
			}
            //NMinusStemp.push_back(NS);
            
//            if(thread == 0)
            {
				printf("NMinusStemp is updated successfully, counter: %d, thread: %d\n", counter, thread);
			}
				
			counter++;
        }
    }while (gpu_increase(Ntemp, it, Ntemp_size, thread, counter));
    
    *subsets_size = counter;
    
    printf("At the end of gpu_generate2, thread: %d\n", thread);
}

//for(int j=indexomp;j< (counterVec[i] + indexomp) ;j++)			// this is to determine the job for each level
		
__global__ void FindOPT(int *dev_ATE_elm, int *dev_counterVec, int indexomp, int *dev_roundVec, const int T, 
						const int k, const int powK, const int AllTableElemets_size, int *dev_ATE_Csubsets, 
						int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, int *dev_zeroVec, 
						int *dev_ATE_optVector, int *dev_ATE_optVector_size, int *dev_ATE_myOPT, 
						int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, const int i, int *it, 
						int *s, int *NS, const int maxSubsetsSize){		
		int thread = blockDim.x * blockIdx.x + threadIdx.x;
		
		int j = thread + indexomp;
		if (thread < dev_counterVec[i]){
            //vector<vector<int> > Ctemp;
            //vector<vector<int> > NMinusStemp;
            //vector<vector<int> > Cwhole;
            //generate2(AllTableElemets[j].elm,Ctemp,NMinusStemp);
            
//            if (thread == 0)
				printf("Before gpu_generate2, counterVec[%d]: %d, thread: %d\n", i, dev_counterVec[i], thread);
            gpu_generate2(&dev_ATE_elm[j * powK], powK, &dev_ATE_Csubsets[j * maxSubsetsSize * powK], &dev_ATE_NSsubsets[j * maxSubsetsSize * powK],
						  dev_roundVec, T, powK, &it[j * powK], &s[j * powK], &NS[j * powK], &dev_ATE_NSsubsets_size[j], thread);     //dev_ATE_elm_size[j] = Ntemp.size() = powK
			
			__syncthreads();
			
//			if (thread == 0)
				printf("gpu_generate2 completed successfully. NSsubsets_size[%d]: %d, thread: %d\n", j, dev_ATE_NSsubsets_size[j], thread);
//            AllTableElemets[j].NSsubsets=NMinusStemp; 	//ni-si
//            AllTableElemets[j].Csubsets=Ctemp;		//configurations
            
//            for(int h=0;h<AllTableElemets[j].NSsubsets.size();h++)   // looking through subset of NSTableElements[j], 
																	// NSTableElements is the table for all previous OPT. Find all subsets(dependency) of selected job
			int optVecIndex = 0;
//			for(int h=0; h < dev_ATE_NSsubsets_size[j]; h++)
			for(int h=0; h < dev_ATE_NSsubsets_size[j]; h++)
            {
//                if(AllTableElemets[j].NSsubsets[h]==zeroVec)   // if subset is zero Vector , its OPT is 0
				if(gpu_sameVectors(&dev_ATE_NSsubsets[(j * maxSubsetsSize + h) * powK], dev_zeroVec, powK))
                {
                    //AllTableElemets[j].optVector.push_back(0);
                    dev_ATE_optVector[j * powK + optVecIndex] = 0;
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
						
						if (thread == 0 && i == 19)
						{
							printf("level 3 starts. j: %d, optVecIndex: %d\n", j, optVecIndex);
						}
						
                        //AllTableElemets[j].optVector.push_back(AllTableElemets[r].myOPT);
                        dev_ATE_optVector[j * powK + optVecIndex] = dev_ATE_myOPT[r];
                        optVecIndex++;
						dev_ATE_optVector_size[j] = optVecIndex;
						
						
						if (thread == 0 && i == 19)
						{
							printf("level 3 complete. h: %d, r: %d\n", h, r);
						}			
				
                        break;
                    }
                }
            }
			
            int minn = 100000;
            int myOptimalindex;
            //for(int pp=0; pp<AllTableElemets[j].optVector.size();pp++)			// find out the OPT from all dependencies.
            for (int pp = 0; pp < dev_ATE_optVector_size[j]; pp++)
            {
       //         cout << AllTableElemets[j].optVector[pp]<<" ";
//                if(AllTableElemets[j].optVector[pp] < minn)
				//if (thread == 0)
				//	printf("j: %d, thread: %d, AllTableElemets[%d].optVector[%d]: %d\n", j, thread, j, pp, dev_ATE_optVector[j*powK+pp]);
				if (dev_ATE_optVector[j * powK + pp] < minn)
                {
//                    minn=AllTableElemets[j].optVector[pp];
					minn = dev_ATE_optVector[j * powK + pp];
                    myOptimalindex=pp;
                }
            }
          //  cout << endl;
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
    int ii=0;
    int indexomp=0;
    int maxSubsetsSize = 0;
	
	InitGPUData(powK, LongJobs_size, AllTableElemets, zeroVec, roundVec, &counterVec[0], maxSubsetsSize, maxSumValue, counterVec.size());

	cout << ", LongJob size: " << LongJobs_size << ", maxSumValue: " << maxSumValue << endl;
	while (ii < maxSumValue+1)		//number of levels = number of jobs + 1
    {
		int tSize = 32;
		int bSize = 1;
		if (tSize < counterVec[ii]){
			bSize = (tSize + counterVec[ii] - 1) / tSize;
		}
//		std::cout << "counterVec[" << ii << "]: " << counterVec[ii] << ", indexomp: " << indexomp << std::endl;
		FindOPT<<<bSize, tSize>>>(dev_ATE_elm, dev_counterVec, indexomp, dev_roundVec, T, k, powK, 
								  AllTableElemets.size(), dev_ATE_Csubsets, dev_ATE_NSsubsets, 
								  dev_ATE_NSsubsets_size, dev_zeroVec, dev_ATE_optVector, 
								  dev_ATE_optVector_size, dev_ATE_myOPT, dev_ATE_myOptimalindex, 
								  dev_ATE_myMinNSVector, ii, it, ss, NS, maxSubsetsSize);
           
//        gpuErrchk(cudaMemcpy(&counterVec[0], dev_counterVec, (LongJobs_size + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        indexomp+=counterVec[ii];
        ii++;
    } 
    
//GPU code to update AllTableElement
	int *temp_NSsubsets, *temp_Csubsets, *temp_myOPT, *temp_myOptIndex, *temp_myMinNSVector, *temp_optVector;
	temp_NSsubsets = new int[AllTableElemets.size() * maxSubsetsSize * powK];
	temp_Csubsets = new int[AllTableElemets.size() * maxSubsetsSize * powK];
	temp_myOPT = new int[AllTableElemets.size()];
	temp_myOptIndex = new int[AllTableElemets.size()];
	temp_myMinNSVector = new int[AllTableElemets.size() * powK];
	temp_optVector = new int[AllTableElemets.size() * (maxSumValue + 1)];
	
	cout << "FindOPT recursion is done. Start memcpy from Device to Host." << endl;
	cout << "AllTableSize: " << AllTableElemets.size() << ", maxSubsetsSize: " << maxSubsetsSize << ", powK: " << powK << endl;
	
	gpuErrchk(cudaMemcpy(temp_NSsubsets, dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_Csubsets, dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_myOPT, dev_ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_myOptIndex, dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_myMinNSVector, dev_ATE_myMinNSVector, AllTableElemets.size() * powK * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_optVector, dev_ATE_optVector, AllTableElemets.size() * (maxSumValue + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	
	std::cout << "memcpy from device to host are done, AllTableElemets.size: " << AllTableElemets.size() << std::endl;

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
		AllTableElemets[i].optVector.insert(AllTableElemets[i].optVector.end(), &temp_optVector[i * (maxSumValue + 1)], &temp_optVector[(i + 1) * (maxSumValue + 1)]);
		AllTableElemets[i].myMinNSVector.insert(AllTableElemets[i].myMinNSVector.end(), &temp_myMinNSVector[i * powK], &temp_myMinNSVector[(i + 1) * powK]);
	}
	
	
	free(temp_NSsubsets);
	free(temp_Csubsets);
	free(temp_myOPT);
	free(temp_myOptIndex);
	free(temp_myMinNSVector);
	free(temp_optVector);
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
}
