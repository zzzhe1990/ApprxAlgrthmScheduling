#ifndef PARALLEL_PTAS
#define PARALLEL_PTAS

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <string>
#include <unistd.h>
#include <sys/time.h>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

class GpuDynamicTable
{
public:
	int *elm;
  thrust::device_vector<int*> NSsubsets;
  thrust::device_vector<int*> Csubsets;
  thrust::device_vector<int> optVector;
	int myOPT;
  int mySum;
  int myOptimalindex;
  int *myMinNSVector;
  bool operator< (const GpuDynamicTable &a) const {
      return (mySum < a.mySum);
  }
};

class DynamicTable
{
public:
	vector <int> elm;
	vector < vector <int> > NSsubsets;
	vector < vector <int> > Csubsets;
	vector <int> optVector;
	//vector <int> CsumOverVec;
	int myOPT;
  int mySum;
  int myOptimalindex;
  vector <int> myMinNSVector;
  bool operator< (const DynamicTable &a) const {
      return (mySum < a.mySum);
  }
};

class FinalTableINFO
{
public:
	vector<int> Nvector;
	int OPTtable;
	vector <DynamicTable> NSTableElements;
	vector <DynamicTable> AllTableElemets;
	int Ttable;
	vector <int> optimalValuesVector;
    vector<int> roundVecTable;
    vector<int> ShortJobsTable;
    vector<int> LongJobsTable;
    vector<int> LongRoundJobsTable;
    double roundCriteriaTable;
};


vector <DynamicTable> NSTableElements;
vector <DynamicTable> AllTableElemets;
vector<int> tempOptVector;
vector < FinalTableINFO > AllProbData;

//  Global defination
int optIndex,FinalMakespan;
int k,T,OPT,LB,UB,s,LB0,UB0;
double error;
long Elapsed_total, secondsT, MicroSecondsT;
int nJobs,nMachines,nFile;
int iwhile;
int roundCriteria;
int f,th;
int numShort, numLong;
int Fopt;


//vector< vector <int> > RoundedOptimalSchedule;
//vector< vector <int> > OptimalSchedule;
vector<int> roundVec;
vector<int> ProcTimeJob;
vector<int> ShortJobs;
vector<int> LongJobs;
vector<int> LongRoundJobs;
vector<int> Ntemp;
vector<int> zeroVec;
//vector<int> machineTimes;

// function declarations
bool increase(const vector<int>& nVector, vector<int>& it);
int Pow(int x, int p);
int sumFun(vector<int> A, vector<int> B);
int DPFunction2(vector<int>& Ntemp);
int rounDownFun(vector<int>& roundVec, vector<int> L,int b);
int Bk(int LB, int UB);
void printFun(vector<int> v);
void printFun(vector<double> v);
void clearFun();
void ListSchedulingFun(vector<int>& ShortJobs,vector<vector<int> >& OptimalSchedule,vector<int>& MachtimeI,int m);
void print(int iwhile);
void findScheduleFun(vector<vector<int> >& FinalMachineConfiguration);
void findScheduleFun2(vector<vector<int> >& FinalMachineConfiguration, vector< vector <int> >& RoundedOptimalSchedule);
void findScheduleFun3(vector<vector<int> >& FinalMachineConfiguration, vector< vector <int> >& RoundedOptimalSchedule);

void generate(vector<int>& Ntemp, vector<vector<int> >& Ctemp, vector<vector<int> >& NMinusStemp,vector<vector<int> >& Cwhole);
void generate2(vector<int>& Ntemp, vector<vector<int> >& Ctemp, vector<vector<int> >& NMinusStemp);
void printFunFile(vector<int> v);
int mainScheduling();
void readInputFile(int nFile);
void printFinalSchedule(vector<vector<int> >& optimalSchedule,vector<int>& Machinetimes, vector<int>& longF,vector<int>& shortF, int Fopt);

void finish_gpu_work();

int nthreads,nthreads0;

//string str0=  "/Users/lalehghalami/Desktop/parallelCode/File";
string str0= "File";

//string str0=  "/wsu/home/ff/ff96/ff9687/ParetoData/File";
//string str0=  "/wsu/home/ff/ff96/ff9687/UniformData/File";
//string str0=  "/Users/lalehghalami/Dropbox/Scheduling/UniformData/File";
string str1;
string str2;
string str3 ="-";
string str4;
string str5=".txt";


ofstream solution("AllInstances.xls");

//ofstream solution("/Users/lalehghalami/Desktop/parallelCode/PTAS-Results-T4F308-5.xls");


//ofstream solution("/wsu/home/ff/ff96/ff9687/Results-Pareto-22Sep/PTAS-Results-T4F308-5.xls");
//ofstream scheduleFile("/wsu/home/ff/ff96/ff9687/Results-8Sep/PTAS-Schedule-T16F40.xls");
//ofstream output("/wsu/home/ff/ff96/ff9687/Results-9Aug/Par-Output-File300.xls");

GpuDynamicTable *h_AllGpuTableElements;
GpuDynamicTable *d_AllGpuTableElements;
int             *h_counterVec;
int							*d_counterVec;

#endif /* PARRALEL_PTAS */
