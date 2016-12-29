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
	
void InitGPUData(int powK, int LongJobs_size, vector<DynamicTable> &AllTableElemets, 
				 int *zeroVec, int *roundVec, int *counterVec, int &maxSubsetsSize)
{
	cout << "At the beginning of InitGPUData. maxSubsetsSize: " << maxSubsetsSize << endl;
	
	int maxIndex = AllTableElemets.size() - 1;
	vector<int> temp;
	cout << "check valid element AllTableElemets[" << maxIndex << "].elm[powK-1]: " << AllTableElemets[maxIndex].elm[powK-1] << endl;
	cout << "This is the current AllTableElemets[" << maxIndex << "].elm: " << endl;
//	for (vector<int>::const_iterator pt = AllTableElemets[maxIndex].elm.end(); pt != AllTableElemets[maxIndex].elm.begin(); --pt)
	for (int p = powK-1; p >= 0; p--)
	{
		cout << AllTableElemets[maxIndex].elm[p] << " ";
		if (AllTableElemets[maxIndex].elm[p] != 0)
			temp.push_back(AllTableElemets[maxIndex].elm[p]);
	}
	cout << endl;
	
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
	
	cout << "AllTableSize: " << AllTableElemets.size() << ", maxSubsetsSize: " << maxSubsetsSize << ", powK: " << powK << ", tempSize: " << temp.size() << endl;
	
	//arrays on device
	gpuErrchk(cudaMalloc((void**)&dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_elm, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myMinNSVector, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myOPT, AllTableElemets.size() * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_optVector, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_counterVec, (LongJobs_size + 1) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_zeroVec, (powK) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_roundVec, (powK) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&it, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&ss, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&NS, AllTableElemets.size() * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_NSsubsets_size, AllTableElemets.size() * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_ATE_optVector_size, AllTableElemets.size() * sizeof(int)));
    
	int *ATE_optVector_size = new int[AllTableElemets.size()];
	int *ATE_myOPT = new int[AllTableElemets.size()];
	for (int i = 0; i < AllTableElemets.size(); i++){
//		ATE_optVector_size[i] = AllTableElemets[i].optVector.size();
		gpuErrchk(cudaMemcpy(&dev_ATE_elm[i * powK], &AllTableElemets[i].elm[0], powK * sizeof(int), cudaMemcpyHostToDevice));
		ATE_myOPT[i] = AllTableElemets[i].myOPT;
	}
	
	gpuErrchk(cudaMemcpy(dev_zeroVec, zeroVec, powK*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_roundVec, roundVec, powK*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_counterVec, counterVec, (LongJobs_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_ATE_optVector_size, ATE_optVector_size, AllTableElemets.size() * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_ATE_myOPT, ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyHostToDevice));
	
	delete(ATE_optVector_size);
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

__device__ int gpu_increase(const int *Ntemp, int *it, int Ntemp_size){
#pragma unroll
	for (int i = 0, size = Ntemp_size; i != size; ++i) {
		const int index = size - 1 - i;
		++it[index];
		if (it[index] > Ntemp[index]) {
			it[index] = 0;
		} else {
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
							  const int T, const int powK, int *it, int *s, int *NS, int *subsets_size){
	
	//Ntemp_size = pow(k,2)
	
	//vector<int> it(Ntemp.size(), 0);
	//int it[Ntemp_size];
	int counter = 0;
#pragma unroll
	for (int i = 0; i < Ntemp_size; i++)
		it[i] = 0;
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
			for (int i = 0; i < Ntemp_size; i++)
			{
				Ctemp[counter*Ntemp_size + i] = s[i];
			}
            //Ctemp.push_back(s);
            
            //int NS[Ntemp_size];
            for(int j=0; j<powK; j++)
            {
                NS[j] = Ntemp[j]-s[j];
            }
            if(gpu_sameVectors(NS, Ntemp, Ntemp_size)){
                continue;
			}
			
			for (int i = 0; i < Ntemp_size; i++)
			{
				NMinusStemp[counter * Ntemp_size + i] = NS[i];
			}
            //NMinusStemp.push_back(NS);
			counter++;
        }
    }while (gpu_increase(Ntemp, it, Ntemp_size));
    
    *subsets_size = counter;
}

//for(int j=indexomp;j< (counterVec[i] + indexomp) ;j++)			// this is to determine the job for each level
		
__global__ void FindOPT(int *dev_ATE_elm, int *dev_counterVec, int indexomp, int *dev_roundVec, const int T, 
						const int k, const int powK, const int AllTableElemets_size, int *dev_ATE_Csubsets, 
						int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, int *dev_zeroVec, 
						int *dev_ATE_optVector, int *dev_ATE_optVector_size, int *dev_ATE_myOPT, 
						int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, const int i, int *it, 
						int *s, int *NS, const int maxSubsetsSize){		
		int thread = blockDim.x * blockIdx.x + threadIdx.x;
//			printf("thread: %d, i: %d, dev_counterVec[i]: %d\n", thread, i, dev_counterVec[i]);
		int j = thread + indexomp;
		if (thread < dev_counterVec[i]){
            //vector<vector<int> > Ctemp;
            //vector<vector<int> > NMinusStemp;
            //vector<vector<int> > Cwhole;
            //generate2(AllTableElemets[j].elm,Ctemp,NMinusStemp);
            
            gpu_generate2(&dev_ATE_elm[j * powK], powK, &dev_ATE_Csubsets[j * maxSubsetsSize * powK], &dev_ATE_NSsubsets[j * maxSubsetsSize * powK],
						  dev_roundVec, T, powK, &it[j * powK], &s[j * powK], &NS[j * powK], &dev_ATE_NSsubsets_size[j]);     //dev_ATE_elm_size[j] = Ntemp.size() = powK
			
			__syncthreads();
//            AllTableElemets[j].NSsubsets=NMinusStemp; 	//ni-si
//            AllTableElemets[j].Csubsets=Ctemp;		//configurations
            
//            for(int h=0;h<AllTableElemets[j].NSsubsets.size();h++)   // looking through subset of NSTableElements[j], 
																	// NSTableElements is the table for all previous OPT. Find all subsets(dependency) of selected job
			int optVecIndex = 0;
//			for(int h=0; h < dev_ATE_NSsubsets_size[j]; h++)
			int hit1 = 0, hit2 = 0, hit3 = 0;
			for(int h=0; h < dev_ATE_NSsubsets_size[j]; h++)
            {
//                if(AllTableElemets[j].NSsubsets[h]==zeroVec)   // if subset is zero Vector , its OPT is 0
				if(gpu_sameVectors(&dev_ATE_NSsubsets[(j * maxSubsetsSize + h) * powK], dev_zeroVec, powK))
                {
                    //AllTableElemets[j].optVector.push_back(0);
                    dev_ATE_optVector[j * powK + optVecIndex] = 0;
                    optVecIndex++;
                    dev_ATE_optVector_size[j] = optVecIndex;
                    hit1++;
                    break;
                }
                //if(AllTableElemets[j].NSsubsets[h]==AllTableElemets[j].elm)   // if NSsubsets[h] is equal to NSTableElements[j] (itself) 
																			//( the one that we are doing operation for it ) ----> break (not interested )
																			// check if it is itself, if yes, ignore OPT of job itself.
				if(gpu_sameVectors(&dev_ATE_NSsubsets[(j * maxSubsetsSize + h) * powK], &dev_ATE_elm[j * powK], powK) ){
                    dev_ATE_optVector_size[j] = optVecIndex;
                    hit2++;
                    break;
				}
                //for(int r=0; r<AllTableElemets.size();r++)        // to find the match in the NSTableElements for reaching OPT
																//dependencies may not be consectively stored in the table, so have to go through the whole
																//table (AllTableElemets) and find them (matched to AllTableElemets[j].NSsubsets[h]).
				for (int r = 0; r < AllTableElemets_size; r++)
                {
//					if (j == 0)
//						printf("thread: %d, j: %d, r: %d, myOPT[%d]: %d\n", thread, j, r, r, dev_ATE_myOPT[r]);
                    //if(AllTableElemets[j].NSsubsets[h]==AllTableElemets[r].elm)   // if found match of NSsubsets[h], copy its OPT and break 
                    if (gpu_sameVectors(&dev_ATE_NSsubsets[(j * maxSubsetsSize + h) * powK], &dev_ATE_elm[r * powK], powK))
                    {
					//	if (thread%100 == 0){
					//		printf("thread: %d, j: %d, NSsubsets seg:", thread, j);
					//		for (int i = 0; i < powK; i++)
					//			printf(" %d", dev_ATE_NSsubsets[(j*maxSubsetsSize+h)*powK+i]);
					//		printf("\n");
					//		printf("thread: %d, j: %d, Csubsets seg:", thread, j);
					//		for (int i = 0; i < powK; i++)
					//			printf(" %d", dev_ATE_elm[r*powK+i]);
					//		printf("\n");
					//	}
                        //AllTableElemets[j].optVector.push_back(AllTableElemets[r].myOPT);
                        dev_ATE_optVector[j * powK + optVecIndex] = dev_ATE_myOPT[r];
                        optVecIndex++;
						dev_ATE_optVector_size[j] = optVecIndex;
						hit3++;
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
	
	InitGPUData(powK, LongJobs_size, AllTableElemets, zeroVec, roundVec, &counterVec[0], maxSubsetsSize);
	
//	for (int i = 0; i < maxSumValue+1; i++){
//		std::cout << "counterVec[" << i << "]: " << counterVec[i] << std::endl;
//	}
	cout << ", LongJob size: " << LongJobs_size << ", maxSumValue: " << maxSumValue << endl;
	while (ii < maxSumValue+1)		//number of levels = number of jobs + 1
    {
		int tSize = 32;
		int bSize = 1;
		if (tSize < counterVec[ii]){
			bSize = (tSize + counterVec[ii] - 1) / tSize;
		}
		//std::cout << "counterVec[" << ii << "]: " << counterVec[ii] << ", bSize: " << bSize << ", tSize: " << tSize << std::endl;
		FindOPT<<<bSize, tSize>>>(dev_ATE_elm, dev_counterVec, indexomp, dev_roundVec, T, k, powK, 
								  AllTableElemets.size(), dev_ATE_Csubsets, dev_ATE_NSsubsets, 
								  dev_ATE_NSsubsets_size, dev_zeroVec, dev_ATE_optVector, 
								  dev_ATE_optVector_size, dev_ATE_myOPT, dev_ATE_myOptimalindex, 
								  dev_ATE_myMinNSVector, ii, it, ss, NS, maxSubsetsSize);
           
//        gpuErrchk(cudaMemcpy(&counterVec[0], dev_counterVec, (LongJobs_size + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        indexomp+=counterVec[ii];
        ii++;
    }
    cudaDeviceSynchronize();
//GPU code to update AllTableElement
	int *temp_NSsubsets, *temp_Csubsets, *temp_myOPT, *temp_myOptIndex;
	temp_NSsubsets = new int[AllTableElemets.size() * maxSubsetsSize * powK];
	temp_Csubsets = new int[AllTableElemets.size() * maxSubsetsSize * powK];
	temp_myOPT = new int[AllTableElemets.size()];
	temp_myOptIndex = new int[AllTableElemets.size()];
	
	cout << "FindOPT recursion is done. Start memcpy from Device to Host." << endl;
	cout << "AllTableSize: " << AllTableElemets.size() << ", maxSubsetsSize: " << maxSubsetsSize << ", powK: " << powK << endl;
	
	gpuErrchk(cudaMemcpy(temp_NSsubsets, dev_ATE_NSsubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_Csubsets, dev_ATE_Csubsets, AllTableElemets.size() * maxSubsetsSize * powK * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_myOPT, dev_ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_myOptIndex, dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost));
	
	std::cout << "memcpy from device to host are done, AllTableElemets.size: " << AllTableElemets.size() << std::endl;
/*	
	cout << "myOPT: ";
	for (int i = 0; i < AllTableElemets.size(); i++){
		cout << temp3[i] << ", ";
	}
	cout << endl;
	
	cout << "NSsubsets of AllTableElemets[0]: ";
	for (int i = 0; i < maxSubsetsSize * powK; i++){
		cout << temp1[i] << ", ";
	}
	cout << endl << "Csubsets of AllTableElemets[0]: ";
	for (int i = 0; i < maxSubsetsSize * powK; i++) {
		cout << temp2[i] << ", ";
	}
	cout << endl;
*/	
	for (int i = 0; i < AllTableElemets.size(); i++)
	{
		AllTableElemets[i].myOPT = temp_myOPT[i];
		AllTableElemets[i].myOptimalindex = temp_myOptIndex[i];
		AllTableElemets[i].NSsubsets.resize(maxSubsetsSize);
		AllTableElemets[i].Csubsets.resize(maxSubsetsSize);
		AllTableElemets[i].optVector.resize(powK);
		int begin = 0, end = maxSubsetsSize * powK;
		while (begin != end)
		{
			AllTableElemets[i].NSsubsets.push_back(std::vector<int>(&temp_NSsubsets[(i * maxSubsetsSize) * powK], &temp_NSsubsets[(i * maxSubsetsSize + 1) * powK]));
			AllTableElemets[i].Csubsets.push_back(std::vector<int>(&temp_Csubsets[(i * maxSubsetsSize) * powK], &temp_Csubsets[(i * maxSubsetsSize + 1) * powK]));
			begin += powK;
		}
		gpuErrchk(cudaMemcpy(&AllTableElemets[i].optVector[0], &dev_ATE_optVector[i * powK], powK * sizeof(int), cudaMemcpyDeviceToHost));
	}
	
	
	free(temp_NSsubsets);
	free(temp_Csubsets);
	free(temp_myOPT);
	free(temp_myOptIndex);
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
