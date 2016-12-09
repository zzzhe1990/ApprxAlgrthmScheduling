#include "DPCuda.hh"

GpuDynamicTable *h_AllGpuTableElements;
GpuDynamicTable *d_AllGpuTableElements;
int             *h_counterVec;
int							*d_counterVec;

__global__
void gpu_dpFunction(GpuDynamicTable *AllGpuTableElements, int *counterVec, int indexomp, int i)
{
  for(int j = indexomp; j < (counterVec[i] + indexomp) ;j++)
  {
    // vector<vector<int> > Ctemp;
    // vector<vector<int> > NMinusStemp;
    //vector<vector<int> > Cwhole;
    // generate2(AllTableElemets[j].elm,Ctemp,NMinusStemp);
    // AllTableElemets[j].NSsubsets=NMinusStemp;
    // AllTableElemets[j].Csubsets=Ctemp;
    //
    // for(int h=0;h<AllTableElemets[j].NSsubsets.size();h++)   // looking through subset of NSTableElements[j]
    // {
    //     if(AllTableElemets[j].NSsubsets[h]==zeroVec)   // if subset is zero Vector , its OPT is 0
    //     {
    //         AllTableElemets[j].optVector.push_back(0);
    //         break;
    //     }
    //     if(AllTableElemets[j].NSsubsets[h]==AllTableElemets[j].elm)   // if NSsubsets[h] is equal to NSTableElements[j] (itself) ( the one that we are doing operation for it ) ----> break (not interested )
    //         break;
    //     for(int r=0; r<AllTableElemets.size();r++)        // to find the match in the NSTableElements for reaching OPT
    //     {
    //
    //         if(AllTableElemets[j].NSsubsets[h]==AllTableElemets[r].elm)   // if found match of NSsubsets[h], copy its OPT and break
    //         {
    //             AllTableElemets[j].optVector.push_back(AllTableElemets[r].myOPT);
    //             break;
    //         }
    //     }
    // }
    // int minn = 100000;
    // int myOptimalindex;
    // for(int pp=0; pp<AllTableElemets[j].optVector.size();pp++)
    // {
    //     //  cout << AllTableElemets[j].optVector[pp]<<" ";
    //     if(AllTableElemets[j].optVector[pp] < minn)
    //     {
    //         minn=AllTableElemets[j].optVector[pp];
    //         myOptimalindex=pp;
    //
    //     }        gpu_dpFunction<<<0, nthreads0>>>(d_AllGpuTableElements, d_counterVec, indexomp, i);
    // AllTableElemets[j].myOptimalindex=myOptimalindex;
    //
    //
    // if(AllTableElemets[j].NSsubsets.size()>0)
    // {
    //     AllTableElemets[j].myMinNSVector=AllTableElemets[j].NSsubsets[myOptimalindex];
    // }
  }
}

// void func_name(GpuDynamicTable *AllGpuTableElements, int *counterVec, int indexomp, int i)
// {
//   gpu_dpFunction<<<0, nthreads0>>>(AllGpuTableElements, counterVec, indexomp, i);
// }

void free_gpu(vector<DynamicTable> &AllTableElemets, vector<int> &counterVec)
{

}

void init_gpu(vector<DynamicTable> &AllTableElemets, vector<int> &counterVec)
{
  int  AllTableElemets_size;
  int  counterVec_size;

  AllTableElemets_size = AllTableElemets.size();
  counterVec_size = counterVec.size();

  h_AllGpuTableElements = new GpuDynamicTable [AllTableElemets_size];
  h_counterVec = new int[counterVec_size];
  h_counterVec = &(counterVec[0]);

  for (int i = 0; i < AllTableElemets_size; i++)
  {
    h_AllGpuTableElements[i].elm = &(AllTableElemets[i].elm[0]);

    for (size_t j = 0; j < AllTableElemets[i].NSsubsets.size(); j++)
      h_AllGpuTableElements[i].NSsubsets[j] = &(AllTableElemets[i].NSsubsets[j][0]);
    for (size_t j = 0; j < AllTableElemets[i].Csubsets.size(); j++)
      h_AllGpuTableElements[i].Csubsets[j] = &(AllTableElemets[i].Csubsets[j][0]);
    for (size_t j = 0; j < AllTableElemets[i].optVector.size(); j++)
      h_AllGpuTableElements[i].optVector.push_back(AllTableElemets[i].optVector[j]);

  	h_AllGpuTableElements[i].myOPT = AllTableElemets[i].myOPT;
    h_AllGpuTableElements[i].mySum = AllTableElemets[i].mySum;
    h_AllGpuTableElements[i].myOptimalindex = AllTableElemets[i].myOptimalindex;
    h_AllGpuTableElements[i].myMinNSVector = &(AllTableElemets[i].myMinNSVector[0]);
  }

  cudaMalloc((void**) &d_AllGpuTableElements, sizeof(GpuDynamicTable) * AllTableElemets_size);
  cudaMemcpy(d_AllGpuTableElements, h_AllGpuTableElements,
      sizeof(GpuDynamicTable) * AllTableElemets_size, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &d_counterVec, sizeof(int) * counterVec_size);
  cudaMemcpy(d_counterVec, h_counterVec, sizeof(int) * counterVec_size,
    cudaMemcpyHostToDevice);
}
