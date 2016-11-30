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

__device__ int gpu_sumFun(int *A, int *B)
{
	int summ=0.0;
#pragma unroll
	for(int i=0; i<(Pow(k,2)); i++)
	{
		summ= summ + A[i]*B[i];
	}
	return summ;
}

__device__ void gpu_generate2(int *Ntemp, int Ntemp_size, int *Ctemp, int *NMinusStemp, int *dev_roundVec, const int T, const int powk){
	
	//Ntemp_size = pow(k,2)
	
	//vector<int> it(Ntemp.size(), 0);
	int it[Ntemp_size];
	int counter = 0;
#pragma unroll
	for (int i = 0; i < Ntemp_size; i++)
		it[i] = 0;
    do {
        int s[Ntemp_size];
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
        int sSum=gpu_sumFun(s,roundVec);
        if(sSum <= T)
        {
#pragma unroll
			for (int i = 0; i < Ntemp_size; i++)
			{
				Ctemp[counter*Ntemp_size + i] = s[i];
			}
            //Ctemp.push_back(s);
            
            int NS[Ntemp_size];
#pragma unroll
            for(int j=0; j<powK; j++)
            {
                NS[j] = Ntemp[j]-s[j];
            }
            if(gpu_sameVectors(NS, Ntemp, Ntemp_size))
                continue;
#pragma unroll
			for (int i = 0; i < Ntemp_size; i++)
			{
				NMinusStemp[counter * Ntemp_size + i] = NS[i];
			}
            //NMinusStemp.push_back(NS);
        }
        counter++;
    }while (gpu_increase(Ntemp, it, Ntemp_size));
}

//for(int j=indexomp;j< (counterVec[i] + indexomp) ;j++)			// this is to determine the job for each level
		
__global__ void FindOPT(int *dev_ATE_elm, int *dev_counterVec, int indexomp, int *dev_roundVec, const int T, const int k, const int powK, const int dev_AllTableElemets_size,
						int *dev_ATE_Csubsets, int *dev_ATE_NSsubsets, int *dev_ATE_NSsubsets_size, int Cwhole_size, int *dev_zeroVec, int *dev_ATE_optVector, int *dev_ATE_optVector_size,
						int *dev_ATE_myOPT, int *dev_ATE_myOptimalindex, int *dev_ATE_myMinNSVector){		
		int thread = blockDim.x * blockIdx.x + threadIdx.x;
		int j = thread + indexomp;
		if (j < dev_counterVec[i] + indexomp){
            //vector<vector<int> > Ctemp;
            //vector<vector<int> > NMinusStemp;
            //vector<vector<int> > Cwhole;
            //generate2(AllTableElemets[j].elm,Ctemp,NMinusStemp);
            
            gpu_generate2(&dev_ATE_elm[j * powK], powK, dev_ATE_Csubsets, dev_ATE_NSsubsets, dev_roundVec, T, powK);     //dev_ATE_elm_size[j] = Ntemp.size() = powK
//            AllTableElemets[j].NSsubsets=NMinusStemp; 	//ni-si
//            AllTableElemets[j].Csubsets=Ctemp;		//configurations
            
//            for(int h=0;h<AllTableElemets[j].NSsubsets.size();h++)   // looking through subset of NSTableElements[j], 
																	// NSTableElements is the table for all previous OPT. Find all subsets(dependency) of selected job
			int optVecIndex = 0;
			for(int h=0; h < dev_ATE_NSsubsets_size[j]; h++)
            {
//                if(AllTableElemets[j].NSsubsets[h]==zeroVec)   // if subset is zero Vector , its OPT is 0
				if(gpu_sameVectors(dev_ATE_NSsubsets[(j * Cwhole_size + h) * powK], dev_zeroVec, powK))
                {
                    //AllTableElemets[j].optVector.push_back(0);
                    dev_ATE_optVector[j * powK + optVecIndex] = 0;
                    optVecIndex++;
                    break;
                }
                //if(AllTableElemets[j].NSsubsets[h]==AllTableElemets[j].elm)   // if NSsubsets[h] is equal to NSTableElements[j] (itself) 
																			//( the one that we are doing operation for it ) ----> break (not interested )
																			// check if it is itself, if yes, ignore OPT of job itself.
				if(gpu_sameVectors(dev_ATE_NSsubsets[(j * Cwhole_size + h) * powK], dev_ATE_elm[j * powK], powK) )
                    break;
                //for(int r=0; r<AllTableElemets.size();r++)        // to find the match in the NSTableElements for reaching OPT
																//dependencies may not be consectively stored in the table, so have to go through the whole
																//table (AllTableElemets) and find them (matched to AllTableElemets[j].NSsubsets[h]).
				for (int r = 0; r < dev_AllTableElemets_size; r++)
                {

                    //if(AllTableElemets[j].NSsubsets[h]==AllTableElemets[r].elm)   // if found match of NSsubsets[h], copy its OPT and break 
                    if (dev_ATE_NSsubsets[(j * Cwhole_size + h) * powK], dev_ATE_elm[r * powK], powK)
                    {
                        //AllTableElemets[j].optVector.push_back(AllTableElemets[r].myOPT);
                        dev_ATE_optVector[j * powK + optVecIndex] = dev_ATE_myOPT[r];
                        optVecIndex++;
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
				// dev_ATE_NSsubsets_size[j], does this vector store all same value for sizes?
				cudaMemcpy(&dev_ATE_myMinNSVector[j * vectorSize], &dev_ATE_NSsubsets[(j * Cwhole_size + myOptimalindex) * powK], powK, cudaMemcpyDeviceToDevice);
				//dev_ATE_myMinNSVector[j] = dev_ATE_NSsubsets[(j * CWhole.SIZE + myOptimalindex) * pow(k,2)];
            }
		}//end if (j)
	}//end FindOPT()
