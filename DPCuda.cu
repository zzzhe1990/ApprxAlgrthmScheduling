#include "DPCuda.hh"

GpuDynamicTable *h_AllGpuTableElements;
GpuDynamicTable *d_AllGpuTableElements;
int             *h_counterVec;
int							*d_counterVec;
int             AllTableElemets_size;
int             counterVec_size;

__global__
void gpu_dpFunction(GpuDynamicTable *AllGpuTableElements, int *counterVec, int indexomp, int i)
{
  for(int j = indexomp; j < (counterVec[i] + indexomp) ;j++)
  {
    // vector<vector<int> > Ctemp;
    // vector<vector<int> > NMinusStemp;
    // vector<vector<int> > Cwhole;
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

void gpu_generate2(/*vector<int>& Ntemp, vector<vector<int> >& Ctemp, vector<vector<int> >& NMinusStemp*/)
{

}

// void func_name(GpuDynamicTable *AllGpuTableElements, int *counterVec, int indexomp, int i)
// {
//   gpu_dpFunction<<<0, nthreads0>>>(AllGpuTableElements, counterVec, indexomp, i);
// }

/*
  Todo :
    - copy d_AllGpuTableElements to h_AllGpuTableElements - OK
    - copy d_counterVec to h_counterVec - OK
    - free h_AllGpuTableElements - OK
    - free h_counterVec - OK
    - free AllTableElemets ?
    - copy h_AllGpuTableElements to AllTableElemets - KO
    - copy h_counterVec to counterVec - KO
*/

void free_gpu(vector<DynamicTable> &AllTableElemets, vector<int> &counterVec)
{
  cudaMemcpy(d_AllGpuTableElements, h_AllGpuTableElements, sizeof(GpuDynamicTable) * AllTableElemets_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(d_counterVec, h_counterVec, sizeof(int) * counterVec_size, cudaMemcpyDeviceToHost);
  cudaFree(d_AllGpuTableElements);
  cudaFree(d_counterVec);
  AllTableElemets.clear(); // destroy elements but don't free the memory
  counterVec.clear();

  for (int i = 0; i < AllTableElemets_size; i++)
  {
    DynamicTable *tmpTable = new DynamicTable();

    tmpTable->elm.assign(h_AllGpuTableElements[i].elm, h_AllGpuTableElements[i].elm + h_AllGpuTableElements[i].elm_size);

    // for (size_t j = 0; j < h_AllGpuTableElements[i].NSsubsets_size; j++) {
    //   for (size_t k = 0; k < h_AllGpuTableElements[i].NSsubsets[j].size(); k++) {
    //     tmpTable->NSsubsets[j].push_back(*h_AllGpuTableElements[i].NSsubsets[j][k].data().get());
    //   }
    // }
    // std::cout << "---------------------------" << '\n';
    int tmp;
    // std::cout << "Csubsets size = " << h_AllGpuTableElements[i].Csubsets_size << '\n';
    for (size_t j = 0; j < h_AllGpuTableElements[i].Csubsets_size ; j++) {
      std::cout << h_AllGpuTableElements[j].Csubsets[j].size() << std::endl;
      for (size_t k = 0; k < h_AllGpuTableElements[j].Csubsets[j].size(); k++) {
        tmp = h_AllGpuTableElements[i].Csubsets[j][k];
        tmpTable->Csubsets[j].push_back(tmp);
        std::cout << "|" << tmpTable->Csubsets[j][k] << ' ';
      }
    }
    std::cout << '\n';

    for (size_t j = 0; j < AllTableElemets[i].optVector.size(); j++)
      h_AllGpuTableElements[i].optVector.push_back(AllTableElemets[i].optVector[j]);

    tmpTable->myOPT = h_AllGpuTableElements[i].myOPT;
    tmpTable->mySum = h_AllGpuTableElements[i].mySum;
    tmpTable->myOptimalindex = h_AllGpuTableElements[i].myOptimalindex;
    // h_AllGpuTableElements[i].myMinNSVector = &(AllTableElemets[i].myMinNSVector[0]);

    AllTableElemets.push_back(*tmpTable);
    delete tmpTable;
  }
  counterVec.assign(h_counterVec, h_counterVec + counterVec_size);
}

void init_gpu(vector<DynamicTable> &AllTableElemets, vector<int> &counterVec)
{
  AllTableElemets_size = AllTableElemets.size();
  counterVec_size = counterVec.size();

  h_AllGpuTableElements = new GpuDynamicTable [AllTableElemets_size];
  h_counterVec = new int[counterVec_size];
  h_counterVec = &(counterVec[0]);

  for (int i = 0; i < AllTableElemets_size; i++)
  {
    h_AllGpuTableElements[i].elm = new int[AllTableElemets[i].elm.size()];
    h_AllGpuTableElements[i].NSsubsets = new thrust::device_vector<int>[AllTableElemets[i].NSsubsets.size()];
    h_AllGpuTableElements[i].Csubsets = new thrust::device_vector<int>[AllTableElemets[i].Csubsets.size()];
    for (size_t j = 0; j < AllTableElemets[i].elm.size(); j++) {
      h_AllGpuTableElements[i].elm[j] = AllTableElemets[i].elm[j];
    }

    for (size_t j = 0; j < AllTableElemets[i].NSsubsets.size(); j++) {
      h_AllGpuTableElements[i].NSsubsets[j] = AllTableElemets[i].NSsubsets[j];
    }
    for (size_t j = 0; j < AllTableElemets[i].Csubsets.size(); j++) {
      h_AllGpuTableElements[i].Csubsets[j] = AllTableElemets[i].Csubsets[j];
    }
    for (size_t j = 0; j < AllTableElemets[i].optVector.size(); j++) {
      h_AllGpuTableElements[i].optVector.push_back(AllTableElemets[i].optVector[j]);
    }

    h_AllGpuTableElements[i].elm_size = AllTableElemets[i].elm.size();
    h_AllGpuTableElements[i].NSsubsets_size = AllTableElemets[i].NSsubsets.size();
    h_AllGpuTableElements[i].Csubsets_size = AllTableElemets[i].Csubsets.size();
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
