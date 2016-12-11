#include "DPCuda.hh"

GpuDynamicTable *h_AllGpuTableElements;
GpuDynamicTable *d_AllGpuTableElements;
int             AllTableElemets_size;

int             *h_counterVec;
int							*d_counterVec;
int             counterVec_size;

int             *h_roundVec;
int             *d_roundVec;
int             roundVec_size;

__device__
void gpu_sumFun(int *A, int *B , int array_size, int *sum, int d_k)
{
  *sum = 0;

  for (size_t i = 0; i < powf((float)d_k, 2.0); i++)
  {
    *sum = *sum + (A[i] * B[i]);
  }
	// int summ=0.0;
	// for(int i=0; i<(Pow(k,2)); i++)
	// {
	// 	summ= summ + A[i]*B[i];
	// }
	// return summ;
}

__device__
void gpu_increase(const int *Ntemp, int *it, int array_size, bool *result)
{
  for (int i = 0; i < array_size; ++i)
  {
    const int index = array_size - 1 - i;
    ++it[index];
    if (it[index] > Ntemp[index])
    {
      it[index] = 0;
    }
    else {
      *result = true;
      break;
    }
  }
  *result = false;
	// for (int i = 0, size = it.size(); i != size; ++i) {
	// 	const int index = size - 1 - i;
	// 	++it[index];
	// 	if (it[index] > Ntemp[index]) {
	// 		it[index] = 0;
	// 	} else {
	// 		return true;
	// 	}
	// }
	// return false;
}

template<typename T>
__device__ T *gpu_realloc(int oldsize, int newsize, T *old)
{
    T* newT = (T*)malloc(newsize * sizeof(T));

    memcpy(newT, old, oldsize);
    free(old);
    return newT;
}

template<typename T>
__device__ void gpu_push_back(T **array, T *elem, int *size)
{
  (*size)++;
  *array = gpu_realloc<T>((sizeof(T) * (*size - 1)), sizeof(T) * (*size), *array);
	memcpy(*array + (*size - 1), (void*)(elem), sizeof(T));
}

__device__
void gpu_generate2(int *Ntemp, int Ntemp_size, thrust::device_vector<int> *Ctemp, int *Ctemp_Size,
                  thrust::device_vector<int> *NMinusStemp, int d_k, int d_T, int *d_roundVec, thrust::device_vector<int> *tmp_s)
{
  int   *it = new int[Ntemp_size];
  bool  result;
  int   powK;

  powK = powf((float)d_k, 2.0);
  result = true;
  memset(it, 0, Ntemp_size);
  do {
    int *s = new int[Ntemp_size];
    int sSum;

    sSum = 0;
    memset(s, 0, Ntemp_size);
    gpu_sumFun(s, d_roundVec, Ntemp_size, &sSum, d_k);
    if (sSum <= d_T)
    {
      thrust::device_vector<int> *tmp;
      thrust::device_vector<int> *NS;
      int NS_size = 0;

      tmp = (thrust::device_vector<int> *)malloc(sizeof(thrust::device_vector<int>));
      NS = (thrust::device_vector<int> *)malloc(sizeof(thrust::device_vector<int>));
      gpu_push_back<thrust::device_vector<int> >(&Ctemp, tmp, Ctemp_Size);

      for (int j = 0; j < powK; j++)
      {
        // int to_push;
        thrust::device_vector<int> *to_push;

        to_push = (thrust::device_vector<int> *)malloc(sizeof(thrust::device_vector<int>));
        // to_push = ;
        memcpy((void*)&to_push[j],
          (void*)(Ntemp[j] - s[j]),
          sizeof(thrust::device_vector<int>));
        printf("Ici -> %d\n", Ntemp[j] - s[j]);
        // gpu_push_back<thrust::device_vector<int> *>(&NS, &to_push, NS_size);
      }
      // if (NS == Ntemp)
      //   continue;
      //
      // gpu_push_back<thrust::device_vector<int> >(&NMinusStemp, NS, NMinusStemp.size());
      // free (tmp);

      // Ctemp.push_back(s);
      // vector<int> NS;
      // for(int j=0; j<Pow(k,2); j++)
      // {
      //     NS.push_back( Ntemp[j]-s[j]);
      // }
      // if(NS==Ntemp)
      //     continue;
      // NMinusStemp.push_back(NS);
    }

    gpu_increase(Ntemp, it, Ntemp_size, &result);
    free (s);
  } while (result);

  // vector<int> it(Ntemp.size(), 0);
  // do {
  //     vector<int> s;
  //     for(vector<int>::const_iterator i = it.begin(); i != it.end(); ++i)
  //     {
  //         s.push_back(*i);
  //     }
  //     //Cwhole.push_back(s);
  //     int sSum=sumFun(s,roundVec);
  //     if(sSum <= T)
  //     {
  //            Ctemp.push_back(s);
  //
  //         vector<int> NS;
  //         for(int j=0; j<Pow(k,2); j++)
  //         {
  //             NS.push_back( Ntemp[j]-s[j]);
  //         }
  //         if(NS==Ntemp)
  //             continue;
  //         NMinusStemp.push_back(NS);
  //     }
  // }while (increase(Ntemp, it));
}

__global__
void gpu_dpFunction(GpuDynamicTable *AllGpuTableElements, int *counterVec, int indexomp,
                  int i, int d_k, int d_T, int *d_roundVec)
{
  // for(int j = indexomp; j < (counterVec[i] + indexomp) ;j++)
  int threadId = threadIdx.x + indexomp;

  if (threadId < (counterVec[i] + indexomp))
  {
    // thrust::device_vector<int>  *Ctemp;
    // thrust::device_vector<int>  *NMinusStemp;
    // thrust::device_vector<int>  *Cwhole;
    // thrust::device_vector<int>  *tmp_s;
    // int Ctemp_Size = 0;
    // int NMinusStemp_Size = 0;
    // int Cwhole_Size = 0;
    //
    // Ctemp = (thrust::device_vector<int>*)malloc(sizeof(thrust::device_vector<int>*) * Ctemp_Size);
    // NMinusStemp = (thrust::device_vector<int>*)malloc(sizeof(thrust::device_vector<int>*) * NMinusStemp_Size);
    // Cwhole = (thrust::device_vector<int>*)malloc(sizeof(thrust::device_vector<int>*) * Cwhole_Size);
    //
    // gpu_generate2(AllGpuTableElements[threadId].elm, AllGpuTableElements[threadId].elm_size, Ctemp, &Ctemp_Size,
    //   NMinusStemp, d_k, d_T, d_roundVec, tmp_s);
    //
    // AllGpuTableElements[threadId].NSsubsets = NMinusStemp; // not sure if memcpy needed ?
    // AllGpuTableElements[threadId].Csubsets = Ctemp;

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

void old_dpFunction(GpuDynamicTable *AllGpuTableElements, int *counterVec, int indexomp, int i)
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

void call_gpu_dpFunction(int indexomp, int i, int k, int T)
{
  gpu_dpFunction<<<1, AllTableElemets_size>>>(
    d_AllGpuTableElements, d_counterVec,
    indexomp, i, k, T, d_roundVec);
}

void free_gpu(vector<DynamicTable> &AllTableElemets, vector<int> &counterVec, vector<int> &roundVec)
{
  int tmp;

  cudaMemcpy(h_AllGpuTableElements, d_AllGpuTableElements, sizeof(GpuDynamicTable) * AllTableElemets_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_counterVec, d_counterVec, sizeof(int) * counterVec_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_roundVec, d_roundVec, sizeof(int) * roundVec.size(), cudaMemcpyDeviceToHost);
  cudaFree(d_AllGpuTableElements);
  cudaFree(d_counterVec);
  cudaFree(d_roundVec);

  AllTableElemets.clear(); // destroy elements but don't free the memory
  counterVec.clear();
  roundVec.clear();
  for (int i = 0; i < AllTableElemets_size; i++)
  {
    DynamicTable *tmpTable = new DynamicTable();

    tmpTable->elm.assign(h_AllGpuTableElements[i].elm, h_AllGpuTableElements[i].elm + h_AllGpuTableElements[i].elm_size);

    for (size_t j = 0; j < h_AllGpuTableElements[i].NSsubsets_size; j++) {
      for (size_t k = 0; k < h_AllGpuTableElements[i].NSsubsets[j].size(); k++) {
        tmp = h_AllGpuTableElements[i].NSsubsets[j][k];
        tmpTable->Csubsets[j].push_back(tmp);
      }
    }
    for (size_t j = 0; j < h_AllGpuTableElements[i].Csubsets_size ; j++) {
      for (size_t k = 0; k < h_AllGpuTableElements[j].Csubsets[j].size(); k++) {
        tmp = h_AllGpuTableElements[i].Csubsets[j][k];
        tmpTable->Csubsets[j].push_back(tmp);
      }
    }

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
  roundVec.assign(h_roundVec, h_roundVec + roundVec_size);
}

void init_gpu(vector<DynamicTable> &AllTableElemets, vector<int> &counterVec, vector<int> &roundVec)
{
  AllTableElemets_size = AllTableElemets.size();
  counterVec_size = counterVec.size();
  roundVec_size = roundVec.size();

  h_AllGpuTableElements = new GpuDynamicTable [AllTableElemets_size];
  h_counterVec = new int[counterVec_size];
  h_counterVec = &(counterVec[0]);
  h_roundVec = &(roundVec[0]);

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
  cudaMalloc((void**) &d_counterVec, sizeof(int) * counterVec_size);
  cudaMalloc((void**)&d_roundVec, sizeof(int) * roundVec.size());

  cudaMemcpy(d_AllGpuTableElements, h_AllGpuTableElements, sizeof(GpuDynamicTable) * AllTableElemets_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_counterVec, h_counterVec, sizeof(int) * counterVec_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_roundVec, h_roundVec, sizeof(int) * roundVec.size(), cudaMemcpyHostToDevice);
}
