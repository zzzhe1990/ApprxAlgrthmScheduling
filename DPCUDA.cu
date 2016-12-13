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
	
void InitGPUData(int AllTableElemets_size, int Cwhole_size, int powK, int LongJobs_size, 
				 vector<DynamicTable> &AllTableElemets, int *zeroVec, int *roundVec, int *counterVec)
{
	//arrays on device
	gpuErrchk(cudaMalloc((void**)&dev_ATE_Csubsets, AllTableElemets_size * Cwhole_size * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_NSsubsets, AllTableElemets_size * Cwhole_size * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_elm, AllTableElemets_size * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myMinNSVector, AllTableElemets_size * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myOPT, AllTableElemets_size * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_myOptimalindex, AllTableElemets_size * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_optVector, AllTableElemets_size * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_counterVec, (LongJobs_size + 1) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_zeroVec, (powK) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_roundVec, (powK) * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&it, AllTableElemets_size * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&ss, AllTableElemets_size * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&NS, AllTableElemets_size * powK * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_ATE_NSsubsets_size, AllTableElemets_size * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_ATE_optVector_size, AllTableElemets_size * sizeof(int)));
    
	int *ATE_NSsubsets_size = new int[AllTableElemets_size];
	int *ATE_optVector_size = new int[AllTableElemets_size];
	for (int i = 0; i < AllTableElemets_size; i++){
//		ATE_NSsubsets_size[i] = AllTableElemets[i].NSsubsets.size();
//		ATE_optVector_size[i] = AllTableElemets[i].optVector.size();
		gpuErrchk(cudaMemcpy(&dev_ATE_elm[i * powK], &AllTableElemets[i].elm[0], powK * sizeof(int), cudaMemcpyHostToDevice));
	}
    
	gpuErrchk(cudaMemcpy(dev_zeroVec, zeroVec, powK*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_roundVec, roundVec, powK*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_counterVec, counterVec, (LongJobs_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_ATE_NSsubsets_size, ATE_NSsubsets_size, AllTableElemets_size * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_ATE_optVector_size, ATE_optVector_size, AllTableElemets_size * sizeof(int), cudaMemcpyHostToDevice));
	
	delete(ATE_NSsubsets_size);
	delete(ATE_optVector_size);
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
							  const int T, const int powK, int *it, int *s, int *NS, int *NSsubsets_size){
	
	//Ntemp_size = pow(k,2)
	
	//vector<int> it(Ntemp.size(), 0);
	//int it[Ntemp_size];
	int counter = 0;
	int NS_counter = 0;
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
#pragma unroll
			for (int i = 0; i < Ntemp_size; i++)
			{
				Ctemp[counter*Ntemp_size + i] = s[i];
			}
            //Ctemp.push_back(s);
            
            //int NS[Ntemp_size];
#pragma unroll
            for(int j=0; j<powK; j++)
            {
                NS[j] = Ntemp[j]-s[j];
            }
            if(gpu_sameVectors(NS, Ntemp, Ntemp_size)){
				for (int i = 0; i < Ntemp_size; i++)
				{
					NMinusStemp[counter * Ntemp_size + i] = -1;
				}
                continue;
			}
#pragma unroll
			for (int i = 0; i < Ntemp_size; i++)
			{
				NMinusStemp[counter * Ntemp_size + i] = NS[i];
				NS_counter++;
			}
            //NMinusStemp.push_back(NS);
        }
        counter++;
    }while (gpu_increase(Ntemp, it, Ntemp_size));
    
    NSsubsets_size[0] = NS_counter;
}

//for(int j=indexomp;j< (counterVec[i] + indexomp) ;j++)			// this is to determine the job for each level
		
__global__ void FindOPT(int *dev_ATE_elm, int *dev_counterVec, int indexomp, int *dev_roundVec, const int T, 
						const int k, const int powK, const int dev_AllTableElemets_size, int *dev_ATE_Csubsets, 
						int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, int Cwhole_size, int *dev_zeroVec, 
						int *dev_ATE_optVector, int *dev_ATE_optVector_size, int *dev_ATE_myOPT, 
						int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, const int i, int *it, 
						int *s, int *NS){		
		int thread = blockDim.x * blockIdx.x + threadIdx.x;
//			printf("thread: %d, i: %d, dev_counterVec[i]: %d\n", thread, i, dev_counterVec[i]);
		int j = thread + indexomp;
		if (thread < dev_counterVec[i]){
            //vector<vector<int> > Ctemp;
            //vector<vector<int> > NMinusStemp;
            //vector<vector<int> > Cwhole;
            //generate2(AllTableElemets[j].elm,Ctemp,NMinusStemp);
            
            gpu_generate2(&dev_ATE_elm[j * powK], powK, &dev_ATE_Csubsets[j * Cwhole_size * powK], &dev_ATE_NSsubsets[j * Cwhole_size * powK],
						  dev_roundVec, T, powK, &it[j * powK], &s[j * powK], &NS[j * powK], &dev_ATE_NSsubsets_size[j]);     //dev_ATE_elm_size[j] = Ntemp.size() = powK
//            AllTableElemets[j].NSsubsets=NMinusStemp; 	//ni-si
//            AllTableElemets[j].Csubsets=Ctemp;		//configurations
            
//            for(int h=0;h<AllTableElemets[j].NSsubsets.size();h++)   // looking through subset of NSTableElements[j], 
																	// NSTableElements is the table for all previous OPT. Find all subsets(dependency) of selected job
			int optVecIndex = 0;
//			for(int h=0; h < dev_ATE_NSsubsets_size[j]; h++)
			for(int h=0; h < Cwhole_size; h++)
            {
//                if(AllTableElemets[j].NSsubsets[h]==zeroVec)   // if subset is zero Vector , its OPT is 0
				if(gpu_sameVectors(&dev_ATE_NSsubsets[(j * Cwhole_size + h) * powK], dev_zeroVec, powK))
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
				if(gpu_sameVectors(&dev_ATE_NSsubsets[(j * Cwhole_size + h) * powK], &dev_ATE_elm[j * powK], powK) ){
                    dev_ATE_optVector_size[j] = optVecIndex;
                    break;
				}
                //for(int r=0; r<AllTableElemets.size();r++)        // to find the match in the NSTableElements for reaching OPT
																//dependencies may not be consectively stored in the table, so have to go through the whole
																//table (AllTableElemets) and find them (matched to AllTableElemets[j].NSsubsets[h]).
				for (int r = 0; r < dev_AllTableElemets_size; r++)
                {

                    //if(AllTableElemets[j].NSsubsets[h]==AllTableElemets[r].elm)   // if found match of NSsubsets[h], copy its OPT and break 
                    if (gpu_sameVectors(&dev_ATE_NSsubsets[(j * Cwhole_size + h) * powK], &dev_ATE_elm[r * powK], powK))
                    {
                        //AllTableElemets[j].optVector.push_back(AllTableElemets[r].myOPT);
                        dev_ATE_optVector[j * powK + optVecIndex] = dev_ATE_myOPT[r];
                        optVecIndex++;
						dev_ATE_optVector_size[j] = optVecIndex;
                        break;
                    }
                }
            }
            int minn = 100000;
            int myOptimalindex;
            //for(int pp=0; pp<AllTableElemets[j].optVector.size();pp++)			// find out the OPT from all dependencies.
#pragma unroll
            for (int pp = 0; pp < dev_ATE_optVector_size[j]; pp++)
            {
       //         cout << AllTableElemets[j].optVector[pp]<<" ";
//                if(AllTableElemets[j].optVector[pp] < minn)
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
					dev_ATE_myMinNSVector[j * powK + i] = dev_ATE_NSsubsets[(j * Cwhole_size + myOptimalindex) * powK + i];
				}
				//dev_ATE_myMinNSVector[j] = dev_ATE_NSsubsets[(j * CWhole.SIZE + myOptimalindex) * pow(k,2)];
            }
            
		}//end if (j)
	}//end FindOPT()
/*
void gpu_DP(vector<DynamicTable> &AllTableElemets, int *dev_ATE_elm, int *dev_counterVec, int *dev_roundVec, 
			const int T, const int k, const int powK, const int dev_AllTableElemets_size,
			int *dev_ATE_Csubsets, int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, 
			int Cwhole_size, int *dev_zeroVec, int *dev_ATE_optVector, int *dev_ATE_optVector_size,
			int *dev_ATE_myOPT, int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector, 
			int *it, int *s, int *NS, const int maxSumValue, vector<int> &counterVec)*/
void gpu_DP(vector<DynamicTable> &AllTableElemets, const int T, const int k, const int powK, 
			const int dev_AllTableElemets_size, int Cwhole_size, const int maxSumValue, 
			vector<int> &counterVec, const int LongJobs_size)
{
    int ii=0;
    int indexomp=0;

//	for (int i = 0; i < maxSumValue+1; i++){
//		std::cout << "counterVec[" << i << "]: " << counterVec[i] << std::endl;
//	}

	while (ii < maxSumValue+1)		//number of levels = number of jobs + 1
    {
		int tSize = 32;
		int bSize = 1;
		if (tSize < counterVec[ii]){
			bSize = (tSize + counterVec[ii] - 1) / tSize;
		}
		//std::cout << "counterVec[" << ii << "]: " << counterVec[ii] << ", bSize: " << bSize << ", tSize: " << tSize << std::endl;
		FindOPT<<<bSize, tSize>>>(dev_ATE_elm, dev_counterVec, indexomp, dev_roundVec, T, k, powK, 
								  dev_AllTableElemets_size, dev_ATE_Csubsets, dev_ATE_NSsubsets, 
								  dev_ATE_NSsubsets_size, Cwhole_size, dev_zeroVec, dev_ATE_optVector, 
								  dev_ATE_optVector_size, dev_ATE_myOPT, dev_ATE_myOptimalindex, 
								  dev_ATE_myMinNSVector, ii, it, ss, NS);
        
               
        gpuErrchk(cudaMemcpy(&counterVec[0], dev_counterVec, (LongJobs_size + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        indexomp+=counterVec[ii];
        ii++;
    }
    
//GPU code to update AllTableElement
	int *temp1, *temp2, *temp3, *temp4;
	temp1 = new int[AllTableElemets.size() * Cwhole_size * powK];
	temp2 = new int[AllTableElemets.size() * Cwhole_size * powK];
	temp3 = new int[AllTableElemets.size()];
	temp4 = new int[AllTableElemets.size()];
	/*
	for(int i=0; i < AllTableElemets.size(); i++){
		AllTableElemets[i].NSsubsets.resize(Cwhole_size);
		AllTableElemets[i].Csubsets.resize(Cwhole_size);
		AllTableElemets[i].optVector.resize(powK);
		for (int j = 0; j < Cwhole_size; j++){
			AllTableElemets[i].NSsubsets[j].resize(powK);
			AllTableElemets[i].Csubsets[j].resize(powK);
//			cudaMemcpy(&AllTableElemets[i].NSsubsets[j][0], &dev_ATE_NSsubsets[(i * Cwhole_size + j) * powK], powK, cudaMemcpyDeviceToHost);
//			cudaMemcpy(&AllTableElemets[i].Csubsets[j][0], &dev_ATE_Csubsets[(i * Cwhole_size + j) * powK], powK, cudaMemcpyDeviceToHost);
			gpuErrchk(cudaMemcpy(temp1, &dev_ATE_NSsubsets[(i * Cwhole_size + j) * powK], powK * sizeof(int), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(temp2, &dev_ATE_Csubsets[(i * Cwhole_size + j) * powK], powK * sizeof(int), cudaMemcpyDeviceToHost));
//			std::copy(temp1, temp1+powK, std::back_inserter(AllTableElemets[i].NSsubsets));
//			std::copy(temp2, temp2+powK, std::back_inserter(AllTableElemets[i].Csubsets));
//			AllTableElemets[i].NSsubsets[j].insert(AllTableElemets[i].NSsubsets[j].begin(), temp1, temp1+powK);
//			AllTableElemets[i].Csubsets[j].insert(AllTableElemets[i].Csubsets[j].begin(), temp2, temp2+powK);
			for (int p = 0; p < powK; p++){
				AllTableElemets[i].NSsubsets[j][p] = temp1[p];
				AllTableElemets[i].Csubsets[j][p] = temp2[p];
			}
			std::cout << "memcpy of NSsubsets and Csubsets for dimension: " << j << " of AllTableElemets[" << i <<"] is done." << std::endl;
		}
		gpuErrchk(cudaMemcpy(&AllTableElemets[i].optVector[0], &dev_ATE_optVector[i * powK], powK * sizeof(int), cudaMemcpyDeviceToHost));
		//Csubsets[Cwhole.size()][powK]
	}
	*/
	gpuErrchk(cudaMemcpy(temp1, dev_ATE_NSsubsets, AllTableElemets.size() * Cwhole_size * powK * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp2, dev_ATE_Csubsets, AllTableElemets.size() * Cwhole_size * powK * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp3, dev_ATE_myOPT, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp4, dev_ATE_myOptimalindex, AllTableElemets.size() * sizeof(int), cudaMemcpyDeviceToHost));
	
	std::cout << "temp1 temp2 is done, AllTableElemets.size: " << AllTableElemets.size() << std::endl;
	for (int i = 0; i < AllTableElemets.size(); i++)
	{
		AllTableElemets[i].myOPT = temp3[i];
		AllTableElemets[i].myOptimalindex = temp4[i];
		AllTableElemets[i].NSsubsets.resize(Cwhole_size);
		AllTableElemets[i].Csubsets.resize(Cwhole_size);
		AllTableElemets[i].optVector.resize(powK);
		int begin = 0, end = Cwhole_size * powK;
		while (begin != end)
		{
			AllTableElemets[i].NSsubsets.push_back(std::vector<int>(&temp1[(i * Cwhole_size) * powK], &temp1[(i * Cwhole_size + 1) * powK]));
			AllTableElemets[i].Csubsets.push_back(std::vector<int>(&temp2[(i * Cwhole_size) * powK], &temp2[(i * Cwhole_size + 1) * powK]));
			begin += powK;
		}
		gpuErrchk(cudaMemcpy(&AllTableElemets[i].optVector[0], &dev_ATE_optVector[i * powK], powK * sizeof(int), cudaMemcpyDeviceToHost));
	}
	
	
	free(temp1);
	free(temp2);
	free(temp3);
	free(temp4);
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
