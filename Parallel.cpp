//#include "Parallel.h"
#include "DPCUDA.h"


//  Global defination 
vector <DynamicTable> NSTableElements;
vector <DynamicTable> AllTableElemets;
vector<int> tempOptVector;
vector < FinalTableINFO > AllProbData;

/************************************************** multiple copy of global vectors for CUDA kernel Optimization ************************************************************/
int MlBk(int LB, int UB, int &sOPT, vector<DynamicTable>& MulNSTableElements, vector<DynamicTable>& MulAllTableElemets, vector<DynamicTable>& MulDispTableElemets,
		vector<int>& MulShortJobs, vector<int>& MulLongJobs, vector<int>& MulLongRoundJobs, vector<int>& MulNtemp, vector<int>& MulroundVec, vector<int>& MultempOptVector,
		vector<FinalTableINFO>& MulAllProbData);
		
int MlDPFunction2(vector<int>& Ntemp, int roundCriteria, const int T, vector<DynamicTable>& MulNSTableElements, vector<DynamicTable>& MulAllTableElemets, 
					vector<DynamicTable>& MulDispTableElemets, vector<int>& MulShortJobs, vector<int>& MulLongJobs, vector<int>& MulLongRoundJobs, 
					vector<int>& MulNtemp, vector<int>& MulroundVec, vector<int>& MultempOptVector, vector<FinalTableINFO>& MulAllProbData);
					
int DimOffset(int dim, int *offsetArr);
int MulDimOffset(vector<int>& coord, vector<int>& weight);
int MlSplitDim(int tar);
void Mlgenerate(vector<int>& Ntemp, vector<vector<int> >& Ctemp, vector<vector<int> >& NMinusStemp, vector<vector<int> >& Cwhole, const int T, vector<int>& MulroundVec);
void MlclearFun(const int size,  vector <vector<FinalTableINFO> >& MulAllProbData);
bool descSort(const dim &a, const dim &b);
/*****************************************************************************************************************************************************************************/

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
void findScheduleFun3(vector<vector<int> >& FinalMachineConfiguration, vector< vector <int> >& RoundedOptimalSchedule);		//extract OPT schedule from DP.

void generate(vector<int>& Ntemp, vector<vector<int> >& Ctemp, vector<vector<int> >& NMinusStemp,vector<vector<int> >& Cwhole);
void generate2(vector<int>& Ntemp, vector<vector<int> >& Ctemp, vector<vector<int> >& NMinusStemp);
void printFunFile(vector<int> v);
int mainScheduling();
void readInputFile(int nFile);
void printFinalSchedule(vector<vector<int> >& optimalSchedule,vector<int>& Machinetimes, vector<int>& longF,vector<int>& shortF, int Fopt);



//int omp_get_num_threads();
//int omp_get_thread_num();
int nthreads;

//string str0=  "/Users/lalehghalami/Desktop/parallelCode/File";
string str0 = "/home/gomc/Desktop/ApproxAlgorthim/Git/ApprxAlgrthmScheduling/UniformData/File";
//string str0=  "/wsu/home/ff/ff96/ff9687/ParetoData/File";
//string str0=  "/wsu/home/ff/ff96/ff9687/UniformData/File";
//string str0=  "/Users/lalehghalami/Dropbox/Scheduling/UniformData/File";
string str1;
string str2;
string str3 ="-";
string str4;
string str5=".txt";
string strr0 = "/home/gomc/Desktop/ApproxAlgorthim/Git/ApprxAlgrthmScheduling/PTAS_Results/PTAS-Results-K40T";
string strr1;
string strr2 = ".xls";

//ofstream solution("AllInstances.xls");


//ofstream solution("/wsu/home/ff/ff96/ff9687/Results-Pareto-22Sep/PTAS-Results-T4F308-5.xls");
//ofstream scheduleFile("/wsu/home/ff/ff96/ff9687/Results-8Sep/PTAS-Schedule-T16F40.xls");
//ofstream output("/wsu/home/ff/ff96/ff9687/Results-9Aug/Par-Output-File300.xls");

int main(int argc, char* argv[])
{

   if(argc<4)
    {
        cout << "not enough parameters have been passed \n";
        cin.get();
        exit(0);
    }
    else
    {
        error=atof(argv[1]);
        th=atoi(argv[2]);
        nFile=atoi(argv[3]);
        nthreads = atoi(argv[4]);
    }
    
    int maxFile = nFile + 1;
    //nFile=153;
    //error =0.3;
    //th=0;

    //nthreads0= Pow(2,th);
    
    ostringstream convertt1, convertt2;
    convertt1 << nthreads;
    strr1 = strr0;
    strr1.append( convertt1.str() );
    strr1.append("F");
    convertt2 << nFile;
    strr1.append( convertt2.str() );
    strr1.append(strr2);
    
	ofstream solution( strr1.c_str() ); 
    solution << " nFile " << " \t" << "f" << "\t" << " njobs " << "\t" << " nMachines " << "\t" <<" Error  "  << "\t"  << " nThreads " << "\t"<< "LB0"<<"\t"<<"UB0"<<"\t " << "Num Short"<<"\t"<<"Num Long"<<"\t"<<"OPT"<<"\t"<<"makespan" << "\t" <<" Total Wall Clock (ms)"<<   "\t" << "Host Name" << endl;

  char hostname[HOST_NAME_MAX];
    if (! gethostname(hostname, sizeof hostname) == 0)
      perror("gethostname");

	struct timeval begin, end;
	long totaltime;
	gettimeofday(&begin, NULL);
	
    while (nFile < maxFile)
    {
        for (int f=1; f<21; f++)
        {
         //   scheduleFile << "File"<<"\t"<<nFile<<"-"<<f<<endl;
            ostringstream convert;
            convert << nFile;
            str1=str0;
            str2 = convert.str();
            str1.append(str2);
            str1.append(str3);
            ostringstream convert2;
            convert2 << f;
            str4 = convert2.str();
            str1.append(str4);
            str1.append(str5);
            cout<<"Reading File..."<<endl;
            ifstream myfile;
            myfile.open(str1.c_str());
            if(!myfile)
            {
                cout << "Error: file could not be opened"  << endl;
                exit (EXIT_FAILURE);
            }
        
            myfile>>nJobs>>nMachines;
            for(int i=0; i<nJobs; i++)
            {
                int p;
                myfile>> p;
                ProcTimeJob.push_back(p);
            }
        
            cout<<"File "<< nFile <<"-"<<f<< " is read."<<endl;
            cout<<"Number of Jobs : "<< nJobs << endl;
            cout<<"Number of Machines : "<< nMachines << endl;
            cout<<"******************************************************************"<<  endl;
            
            solution<<nFile<<"\t" <<f<<"\t"<< nJobs << "\t" << nMachines << "\t" << error << "\t" ;
        
            struct timeval start_total, end_total;
            gettimeofday(&start_total, NULL);
           
            s=mainScheduling();
            
            gettimeofday(&end_total, NULL);
            secondsT  = end_total.tv_sec  - start_total.tv_sec;
            MicroSecondsT = end_total.tv_usec - start_total.tv_usec;
            Elapsed_total = (secondsT * 1000 + MicroSecondsT / 1000.0) + 0.5;
            
            solution << LB0<<"\t"<<UB0<<"\t"<<numShort<<"\t"<<numLong<<"\t"<<Fopt<<"\t"<< FinalMakespan << "\t"<<Elapsed_total<< "\t"<< hostname << endl;

            
            AllProbData.clear();
            
          //  solution << nthreads << "\t"<< LB0<<"\t"<<UB0<<"\t"<<FinalMakespan << "\t"<<Elapsed_total<< endl;

            cout<<"File "<< nFile <<"-"<<f<< " is Done."<<endl;
            cout << "******************************"<<endl;
        }
        
        nFile++;
    }
    
    gettimeofday(&end, NULL);
    totaltime = end.tv_sec - begin.tv_sec;
    cout << "Total running time is: " << totaltime << endl;

	return 0;
}

int mainScheduling()
{
	int avMachineLoad;
    vector< vector <int> > RoundedOptimalSchedule;
    vector< vector <int> > OptimalSchedule;
    vector <int > shortF;
    vector <int> longF;
    vector<int> machineTimes;
    vector <vector<FinalTableINFO> > MulAllProbData;
    
    
	//Calculating Lower and Upper bounds
	int MaxP=0;
	double sum=0;
	for(int i=0; i< ProcTimeJob.size(); i++)
	{
		sum=sum + ProcTimeJob[i];
		if(ProcTimeJob[i]>MaxP)
			MaxP=ProcTimeJob[i];
	}

	avMachineLoad= ceil(sum/nMachines);
	LB=max(avMachineLoad,MaxP);
	UB=avMachineLoad+MaxP;
    LB0=LB;
    UB0=UB;

	iwhile=1;
    int BkID;
	
    struct timeval tempt, lt, st;
    long iterT, wallClockT;
    clock_t t;
	gettimeofday(&st, NULL);
	
	/**********************************************************/
	/**************** Optimization Variables ******************/
	int numThread = 4;
	int seg = min(4, numThread);		//how many segments to split (threads)
	int *sLB, *sUB, *sT, *sOPT, *sBkID, *dirc;
	sLB = new int[4];
	sUB = new int[4];
	sT = new int[4];
	sOPT = new int[4];
	sBkID = new int[4];
	dirc = new int[4];
	
	for (int x = 0; x < seg; x++)
	{
		MulAllProbData.push_back(vector<FinalTableINFO> ());
	}
	//MulAllProbData.reserve(4);
	
	int marker = 0;
	while(LB<UB){
		gettimeofday(&tempt, NULL);
		t = clock();
		BkID = 0;
		
		marker++;
#ifdef _HOST_DEBUG		
		cout << "Iteration: " << marker << ", UB: " << UB << ", LB: " << LB << "start at time: " << tempt.tv_sec << endl;
#endif	
		//if (UB - LB + 1 >= seg)
		if (UB - LB > seg)
		{
			memset(sOPT, 0, seg);			
			
			//int segSize = ( (UB - LB)/2 + 1)/seg;
			int segSize = (UB - LB + 1)/seg;
			
			for (int i = 0; i < seg; i++)
			{
				sLB[i] = LB + i * segSize;
				//sUB[i] = sLB[i]+segSize - 1;
				sUB[i] = sLB[i]+segSize;
				if (i == seg-1){
					//sUB[i] = max(sLB[i]+segSize-1, UB);
					sUB[i] = max(sLB[i]+segSize, UB);
				}
				sT[i] = sLB[i] + (sUB[i] - sLB[i])/2;
			}
			
			#pragma omp parallel shared(sLB, sUB, sOPT, sBkID, MulAllProbData) num_threads(numThread)	
			{
				vector<DynamicTable> MulNSTableElements;
				vector<DynamicTable> MulAllTableElemets;
				vector<DynamicTable> MulDispTableElemets;
				vector<int> MulShortJobs;
				vector<int> MulLongJobs;
				vector<int> MulLongRoundJobs;
				vector<int> MulNtemp;
				vector<int> MulroundVec;
				vector<int> MultempOptVector;
				#pragma omp for	schedule(static,1)	
				for (int i = 0; i < seg; i++)
				{
					sBkID[i] = MlBk(sLB[i], sUB[i], sOPT[i], MulNSTableElements, MulAllTableElemets, MulDispTableElemets, MulShortJobs, MulLongJobs,
									MulLongRoundJobs, MulNtemp, MulroundVec, MultempOptVector, MulAllProbData[i]);
				}	
#ifdef _HOST_DEBUG
		cout << "Thread: " << omp_get_thread_num() << ", MlBk function is done, and BkID for all processes are collected." << endl;
#endif		
			}
					
			//There might be an error on "break". Require double check at the end.
			for (int i = 0; i < seg; i++)
				BkID += sBkID[i];
				
			if (BkID < 0)
			{
				cout << "something wrong is going wrong here because BkID < 0, process terminates here." << endl;
				break;
			}

			//use sOPT to adjust LB and UB. If all sOPT[] are either larger or smaller than nMachines, LB & UB are updated to the right most or left most seg edges; 
			//If sOPT[i]>nMachine, but sOPT[i+1] < nMachine, then LB = sT[i], UB = sT[i+1]
#ifdef _HOST_DEBUG		
			cout << "sBkID is: ";
			for (int i=0; i<seg; i++)
				cout << sBkID[i] <<" ";
			cout << endl << "BkID: " << BkID << endl;
			cout << "Thread: " << omp_get_thread_num() << " has dirc:";
#endif
			for (int i = 0; i < seg; i++)
			{
				if (sOPT[i] <= nMachines){
					dirc[i] = -1;
				}
				else{
					dirc[i] = 1;
				}
#ifdef _HOST_DEBUG
				cout << " " << dirc[i];
#endif
			}
			cout << endl;
			
			if (dirc[0] == -1){
				LB = sLB[0];
				UB = sT[0]+1;
				OPT = sOPT[0];
				T = (LB + UB)/2;
#ifdef _HOST_DEBUG
				cout << "Pick seg 0 for next iteration where LB: " << LB << ", UB: " << UB << endl;
#endif			
				if (OPT <= nMachines){
					AllProbData.push_back(MulAllProbData[0][0]);
				}
				if (sUB[0]-sLB[0] == 1){
					UB = LB;
				}
			}
			else if(dirc[seg-1] == 1){
				LB = sT[seg-1]+1;
				UB = sUB[seg-1];
				OPT = sOPT[seg-1];
				T = (LB + UB)/2;
#ifdef _HOST_DEBUG
				cout << "Pick seg " << seg-1 << " for next iteration where LB: " << LB << ", UB: " << UB << endl;
#endif				
				if (OPT <= nMachines){
					AllProbData.push_back(MulAllProbData[seg-1][0]);
				}
				if (sUB[seg-1]-sLB[seg-1] == 1){
					LB = UB;
				}
			}
			else{
				for (int i = 1; i < seg; i++){
					if (dirc[i] != dirc[i-1]){
						LB = sT[i-1] + 1;
						UB = sT[i]+1;
						OPT = sOPT[i];
						T = (LB + UB)/2;
#ifdef _HOST_DEBUG
						cout << "Pick seg " << i << " for next iteration where LB: " << LB << ", UB: " << UB << endl;
#endif						
						if (OPT <= nMachines){
							AllProbData.push_back(MulAllProbData[i][0]);
						}
						if (sUB[i-1]-sLB[i-1] == 1 && sUB[i]-sLB[i] == 1){
							UB = sUB[i-1];
							LB = UB;
							T = (sLB[i-1] + sUB[i-1])/2;
						}
						break;
					}
				}
			}
		}
		else{
			memset(sOPT, 0, seg);			
			//seg = UB - LB + 1;		
			seg = UB - LB;
			numThread = min(seg, numThread);
			
			for (int i = 0; i < seg; i++)
			{
				sLB[i] = LB + i;
				//sUB[i] = sLB[i];
				sUB[i] = sLB[i] + 1;
			}
					
			#pragma omp parallel shared(sLB, sUB, sOPT, sBkID, MulAllProbData) num_threads(numThread)	
			{
				vector<DynamicTable> MulNSTableElements;
				vector<DynamicTable> MulAllTableElemets;
				vector<DynamicTable> MulDispTableElemets;
				vector<int> MulShortJobs;
				vector<int> MulLongJobs;
				vector<int> MulLongRoundJobs;
				vector<int> MulNtemp;
				vector<int> MulroundVec;
				vector<int> MultempOptVector;
				#pragma omp for	schedule(static,1)	
				for (int i = 0; i < seg; i++)
				{
					sBkID[i] = MlBk(sLB[i], sUB[i], sOPT[i], MulNSTableElements, MulAllTableElemets, MulDispTableElemets, MulShortJobs, MulLongJobs,
									MulLongRoundJobs, MulNtemp, MulroundVec, MultempOptVector, MulAllProbData[i]);
				}
			}			
					
			//There might be an error on "break". Require double check at the end.
			for (int i = 0; i < seg; i++)
				BkID += sBkID[i];
			if (BkID < 0)
				break;
				
			//use sOPT to adjust LB and UB. If all sOPT[] are either larger or smaller than nMachines, LB & UB are updated to the right most or left most seg edges; 
			for (int i = 0; i < seg; i++)
			{
				if (sOPT[i] <= nMachines){
					dirc[i] = -1;
				}
				else{
					dirc[i] = 1;
				}
			}
						
			if (dirc[0] == -1){
				LB = sLB[0];
				//UB = sUB[0];
				UB = LB;
				OPT = sOPT[0];
				T = (LB + UB)/2;
				
				if (OPT <= nMachines){
					AllProbData.push_back(MulAllProbData[0][0]);
				}
			}
			else if(dirc[seg-1] == 1){
				//LB = sLB[seg-1];
				UB = sUB[seg-1];
				LB = UB;
				OPT = sOPT[seg-1];
				T = (LB + UB)/2;
				
				if (OPT <= nMachines){
					AllProbData.push_back(MulAllProbData[seg-1][0]);
				}
			}
			else{
				for (int i = 1; i < seg; i++){
					if (dirc[i] != dirc[i-1]){
						//LB = sLB[i];
						//UB = sUB[i];
						LB = sLB[i];
						UB = LB;
						OPT = sOPT[i];
						T = (LB + UB)/2;
						
						if (OPT <= nMachines){
							AllProbData.push_back(MulAllProbData[i][0]);
						}
						break;
					}
				}
			}
		}
		
		gettimeofday(&lt, NULL);
		
		iterT = lt.tv_sec - tempt.tv_sec;
		wallClockT = lt.tv_sec - st.tv_sec;
		t = clock() - t;

		cout << "BKID: " << BkID << ", LB: " << LB << ", UB: " << UB << ", OPT: " << OPT << endl;
		cout << "Dynamic Programming Runtime: " << (float)t / CLOCKS_PER_SEC << endl;
		cout << "Execution time between LB and UB is: " << iterT << endl;
		cout << "By far, all LB UB calculation runtime: " << wallClockT << endl;
		iwhile++;
		
		MlclearFun(seg, MulAllProbData);
	}

	gettimeofday(&lt, NULL);
    
    cout << "********************************************************"<<endl;
    cout << "Total execution on UB and LB is: " << lt.tv_sec - st.tv_sec << endl;
    cout << "OUT of Bk while loop  "<<endl;
    cout << "UB    "<< UB<<endl;
    cout << "LB    "<< LB<<endl;
    cout << "T1    "<< T<<endl;
   	double TT;
    TT=(LB+UB)/2;
    T=floor(TT);
    cout << "T2    "<< T<<endl;
    cout << "OPT   "<< OPT<<endl;
    cout << "nMachines   "<< nMachines<<endl;
    cout << "BkID	" << BkID << endl;
    cout << "AllProbData.size()    "<< AllProbData.size()<<endl;
    
    
    for (int ck=AllProbData.size()-1; ck >= 0; ck--)
    {
       cout << "ck :   " << ck << endl;
       cout << "AllProbData[ck].Ttable    "<< AllProbData[ck].Ttable <<endl;

    }

    vector<int> temp;
	for(int i=0;i<nMachines;i++)
	{
		OptimalSchedule.push_back(temp);
	}
	for(int i=0;i<nMachines;i++)
	{
		machineTimes.push_back(0);
	}
    
    
	cout << "No Problem until here 1" << endl;
    
    if (AllProbData.size() != 0){
		if(BkID==0) // number of long jobs are more than 0
		{
			if(OPT<=nMachines)
				optIndex=AllProbData.size()-1;
			else if(OPT>nMachines)
			{
				for (int ck=AllProbData.size()-1; ck >= 0; ck--)
				{
					if (AllProbData[ck].Ttable==T)
					{
						optIndex=ck;

						break;
					}
				}
			}
			Fopt=AllProbData[optIndex].OPTtable;
			longF= AllProbData[optIndex].LongJobsTable;
			shortF=AllProbData[optIndex].ShortJobsTable;
		}else if(BkID==-1) // number of long jobs are zero ... No long jobs
		{
			
			Fopt=0;
			longF=LongJobs;
			shortF=ShortJobs;
	 //       nthreads= -1;
		}
		numShort=shortF.size();
		numLong=longF.size();
		
    
		cout << "No Problem until here 2" << endl;

		if(longF.size()!=0)
		{
			vector< vector <int> > FinalMachineConfiguration;
			vector<int> tempVec;    
			for(int i=0;i<nMachines;i++)
			{
				RoundedOptimalSchedule.push_back(tempVec);
			}
			//findScheduleFun2(FinalMachineConfiguration,RoundedOptimalSchedule);
			findScheduleFun3(FinalMachineConfiguration,RoundedOptimalSchedule);


			// Now, we need to interpret the schedule for rounded down Long jobs as a schedule for Long jobs with original Processing time
			
			for(int i=0;i<RoundedOptimalSchedule.size();i++)
			{
				for(int j=0; j< RoundedOptimalSchedule[i].size();j++)
				{
					for(int a=0;a<longF.size();a++)
					{
						if(RoundedOptimalSchedule[i][j] != AllProbData[optIndex].roundCriteriaTable*Pow(k,2))
						{
							if(RoundedOptimalSchedule[i][j] <= longF[a] && longF[a] < RoundedOptimalSchedule[i][j] +  AllProbData[optIndex].roundCriteriaTable )
							{
								OptimalSchedule[i].push_back(longF[a]);
								longF.erase(longF.begin()+ a);
								break;
							}
						}else
						{
							if(RoundedOptimalSchedule[i][j] <= longF[a]  )
							{
								OptimalSchedule[i].push_back(longF[a]);
								longF.erase(longF.begin()+ a);
								break;
							}
						}
					}				
				}
			}
		
			for(int i=0;i<OptimalSchedule.size();i++)
			{
				int ss=0;
				for(int j=0;j<OptimalSchedule[i].size();j++)
				{
					ss=ss+OptimalSchedule[i][j];
				}
				machineTimes[i]=ss;
			}
			
			
			int makespan;
			makespan = max_element(machineTimes.begin(), machineTimes.end()) - machineTimes.begin();
			
		}
	   
	   
		cout << "No Problem until here 3" << endl;

		if(shortF.size()!=0){
			ListSchedulingFun(shortF,OptimalSchedule,machineTimes,OPT);	
		  //  cout << "Short jobs has been aded"<<endl;
		}


		int makespan;
		makespan = max_element(machineTimes.begin(), machineTimes.end()) - machineTimes.begin();
		FinalMakespan = machineTimes[makespan];
		 
		cout << "Final OptimalSchedule" << endl;
		
		
		cout << "FinalMakespan" << FinalMakespan<< endl;
		printFinalSchedule( OptimalSchedule, machineTimes, longF,shortF,Fopt);
	}

	delete[] sLB;
	delete[] sUB;
	delete[] sT;
	delete[] sOPT;
	delete[] sBkID;
	delete[] dirc;

//	NSTableElements.clear();
//	AllTableElemets.clear();
//	tempOptVector.clear();
	RoundedOptimalSchedule.clear();
	OptimalSchedule.clear();
	roundVec.clear();
	ProcTimeJob.clear();
	ShortJobs.clear();
	LongJobs.clear();
	LongRoundJobs.clear();
	Ntemp.clear();
	machineTimes.clear();
    shortF.clear();
    longF.clear();
    
    cout <<"Main Scheduling is Done"<<endl;
	return 0;
}


int Bk(int LB, int UB)
{

	//calculating T and k
	double TT;
	double creteria;
	TT=(LB+UB)/2;
	T=floor(TT);
	double dk=1/error;
	k=ceil(dk);
	creteria=TT/double(k);

	// Seprating Short and Long Jobs
	for(int i=0; i< ProcTimeJob.size(); i++)
	{
		if(ProcTimeJob[i]<=creteria)
		{
			ShortJobs.push_back(ProcTimeJob[i]);
			
		}
		else
		{
			LongJobs.push_back(ProcTimeJob[i]);
		}
	}

	roundCriteria=(T/(Pow(k,2)));
    

    
	//Generating T/K^2 vector : will be used to generating n vector
	for(int i=0; i<Pow(k,2); i++)
	{
		roundVec.push_back((i+1)*roundCriteria);
	}
    
   
   
	for(int iterator=0; iterator<LongJobs.size();iterator++){
		LongRoundJobs.push_back(rounDownFun(roundVec,LongJobs,iterator));}

	// Generating vector N ( by comparing the rounded down Long Jobs processing times with r vector and counting them
	for(int i=0; i<roundVec.size(); i++)
	{
		int count=0;
		for(int j=0; j<LongRoundJobs.size();j++)
		{
			if(roundVec[i]==LongRoundJobs[j])
				count++;
		}
		Ntemp.push_back(count);
	}
    
    
    //cout << "Ntemp is done as following " <<endl;
   // for (int ck=0; ck<Ntemp.size(); ck++) {
    //    cout << Ntemp[ck]<< "  ";
    //}
    //cout << endl;

	//Starting Dynamic Programing Algorithm
	AllTableElemets.clear();
	NSTableElements.clear();
	tempOptVector.clear();

    
    if (LongJobs.size()>0) {
        //cout << " LongJobs.size() " <<LongJobs.size()<<endl;
        OPT=DPFunction2(Ntemp);
        return 0;
    }
    else
    {
		//cout << " LongJobs.size() " <<LongJobs.size()<<endl;
        return -1;
	}
	
	
}

int MlBk(int LB, int UB, int &sOPT, vector<DynamicTable>& MulNSTableElements, vector<DynamicTable>& MulAllTableElemets, vector<DynamicTable>& MulDispTableElemets,
		vector<int>& MulShortJobs, vector<int>& MulLongJobs, vector<int>& MulLongRoundJobs, vector<int>& MulNtemp, vector<int>& MulroundVec, 
		vector<int>& MultempOptVector, vector<FinalTableINFO>& MulAllProbData)
{
	//calculating T and k
	double TT;
	double creteria;
	int roundCriteria;
	TT=(LB+UB)/2;
	int MlT=floor(TT);
	double dk=1/error;
	k=ceil(dk);
	creteria=TT/double(k);
	
	int thread = omp_get_thread_num();

#ifdef _HOST_DEBUG	
	cout << "thread: " << thread << ", MlBk is called with LB: " << LB << ", UB: " << UB << endl;
#endif

	// Seprating Short and Long Jobs
	for(int i=0; i< ProcTimeJob.size(); i++)
	{
		if(ProcTimeJob[i]<=creteria)
		{
			MulShortJobs.push_back(ProcTimeJob[i]);
			
		}
		else
		{
			MulLongJobs.push_back(ProcTimeJob[i]);
		}
	}

	roundCriteria=(MlT/(Pow(k,2)));
    

    
	//Generating T/K^2 vector : will be used to generating n vector
	for(int i=0; i<Pow(k,2); i++)
	{
		MulroundVec.push_back((i+1)*roundCriteria);
	}
    
   
	for(int iterator=0; iterator<MulLongJobs.size();iterator++){
		MulLongRoundJobs.push_back(rounDownFun(MulroundVec, MulLongJobs, iterator));}

	// Generating vector N ( by comparing the rounded down Long Jobs processing times with r vector and counting them
	for(int i=0; i<MulroundVec.size(); i++)
	{
		int count=0;
		for(int j=0; j<MulLongRoundJobs.size();j++)
		{
			if(MulroundVec[i]==MulLongRoundJobs[j])
				count++;
		}
		MulNtemp.push_back(count);
	}

	//Starting Dynamic Programing Algorithm
	MulAllTableElemets.clear();
	MulDispTableElemets.clear();
	MulNSTableElements.clear();
	MultempOptVector.clear();

    
    if (MulLongJobs.size()>0) {
#ifdef _HOST_DEBUG		
        cout << " thread: " << thread << ", LongJobs.size() " <<MulLongJobs.size()<<endl;
#endif
        sOPT = MlDPFunction2(MulNtemp, roundCriteria, MlT, MulAllTableElemets, MulDispTableElemets, MulNSTableElements, 
							 MulShortJobs, MulLongJobs, MulLongRoundJobs, MulNtemp, MulroundVec, MultempOptVector, MulAllProbData);
#ifdef _HOST_DEBUG
		cout << "thread: " << thread << ", sOPT: " << sOPT << endl;
#endif
        return 0;
    }
    else{
#ifdef _HOST_DEBUG		
		cout << " LongJobs.size() " << MulLongJobs.size()<<endl;
#endif
        return -1;
	}
}


int DPFunction2(vector<int>& Ntemp)
{

    //AllProbData.clear();
	vector<vector<int> > Ctemp;
	vector<vector<int> > NMinusStemp;
	vector<vector<int> > Cwhole;
	generate(Ntemp,Ctemp,NMinusStemp,Cwhole);

	for(int i=0;i<Cwhole.size();i++)
	{
		DynamicTable tempInst1;
		tempInst1.elm=Cwhole[i];
		AllTableElemets.push_back(tempInst1);
	}

	for(int i=NMinusStemp.size()-1;i>-1;i--)
	{
		DynamicTable tempInst;
		tempInst.elm=NMinusStemp[i];
		NSTableElements.push_back(tempInst);
	}

    vector<int> sumVec;
	for (int i=0; i<AllTableElemets.size(); i++) 	//calculate the total run time of each job, jobs have same run time are in the same level (can do parallel).  AllTableElemets.size() = CWhole.size()
	{
		int sumOver=0;
		for (int j=0; j<AllTableElemets[i].elm.size(); j++)   //AllTableElemets[i].elm.size() = Cwhole[i].szie() = Ntemp.size()
		{
			sumOver  = sumOver+ AllTableElemets[i].elm[j];
		}
		sumVec.push_back(sumOver);
        AllTableElemets[i].mySum=sumOver;
	}
    
    sort(sumVec.begin(), sumVec.end());			//align jobs in order for dependency.
    vector<int> counterVec;
    
    // count how many of each item is in the sumVec
    for(int i = 0; i < sumVec.size(); i++)		//the number of jobs in each level       sumVec.size() = AllTableElemets.size()
    {
        int c = 1;
        
        int limit = sumVec.size() - 1;
        while(i < limit  && sumVec[i] == sumVec[i+1])
        {
            c++;
            i++;
        }
        
        counterVec.push_back(c);
    }
    
    sort(AllTableElemets.begin(), AllTableElemets.end());

	int maxSumIndex;
	maxSumIndex = max_element(sumVec.begin(), sumVec.end()) - sumVec.begin();			
	int maxSumValue = sumVec[maxSumIndex];											//the max time among all configuration

	vector<int> zeroVec;
	for(int i=0;i< Ntemp.size();i++)
		zeroVec.push_back(0);
    
    int powK = pow(k,2);
    

    //gpu_DP(AllTableElemets, T, k, powK, maxSumValue, counterVec, LongJobs.size(), &zeroVec[0], &roundVec[0]);
	
	
	for(int i=0; i<NSTableElements.size();i++)			//NSTableElements is N - S. For example. (2,3), si = (0,1), then NS[i] = (2,2)
	{
		for(int j=0;j<AllTableElemets.size();j++)
		{	
			if(NSTableElements[i].elm==AllTableElemets[j].elm)
			{
				NSTableElements[i]=AllTableElemets[j];
				break;
			}
		}

	}

	for(int i=0; i<NSTableElements.size();i++)
	{
		tempOptVector.push_back(NSTableElements[i].myOPT);
	}

	int dpoptimal;
	int minN=100000;
	 
	for(int mindex=0; mindex<NSTableElements.size();mindex++)
	{
		if(NSTableElements[mindex].myOPT<minN)
			minN=NSTableElements[mindex].myOPT;
	}

	dpoptimal=minN +1;

    cout << "dpoptimal: " <<dpoptimal << endl;
    
    clock_t ttt = clock();
    
    
    if (dpoptimal<=nMachines)		//keep a copy of the latest feasible solution.
    {
        
        FinalTableINFO instance;
        instance.AllTableElemets=AllTableElemets;
        instance.NSTableElements=NSTableElements;
        instance.Nvector=Ntemp;
        instance.OPTtable=dpoptimal;
        instance.Ttable=T;
        instance.optimalValuesVector=tempOptVector;
        instance.ShortJobsTable=ShortJobs;
        instance.LongJobsTable=LongJobs;
        instance.roundVecTable=roundVec;
        instance.LongRoundJobsTable=LongRoundJobs;
        instance.roundCriteriaTable=roundCriteria;
        
        AllProbData.push_back(instance);
    }

	ttt = clock() - ttt;
	cout << "copy the latest feasible solution to instance takes time: " << (float)ttt/CLOCKS_PER_SEC << endl;

	sumVec.clear();
	Cwhole.clear();
  
	return dpoptimal;
}


// This is the function to calculate the offset of one dimension
int DimOffset(int dim, vector<int>& dimSize)
{
	int offset = 1;
	    
	for (int i=dim; i<dimSize.size(); i++)   
	{
		offset *= dimSize[i];
	}

	return offset;
}

//This is the function to calculate the offset for multi-dimension configuration
int MulDimOffset(vector<int>& coord, vector<int>& weight)
{
	//For the first dimension, it has no offset but itself. Add to offset directly
	int numDim = coord.size()-1;
	int offset = coord[numDim];
	//For higher dimensions, offset = its index * size of all dimensions that lower than itself. eg: d3 offset = index d3 * d2 size * d1 size * d0 size + d2 * d1 size * d0 size + d1 * d0 size + d0
	for (int i=0; i<numDim; i++)
	{
		offset += ( coord[i] * DimOffset(i+1, weight) ); //i: index of one dimension; numDim -i = dimension id
	}

	return offset;
}

bool descSort(const dim &a, const dim &b)
{
	return (a.weit > b.weit);
}


int MlDPFunction2(vector<int>& Ntemp, int roundCriteria, const int MlT, vector<DynamicTable>& MulNSTableElements, vector<DynamicTable>& MulAllTableElemets, 
					vector<DynamicTable>& MulDispTableElemets, vector<int>& MulShortJobs, vector<int>& MulLongJobs, vector<int>& MulLongRoundJobs, 
					vector<int>& MulNtemp, vector<int>& MulroundVec,  vector<int>& MultempOptVector, vector<FinalTableINFO>& MulAllProbData)
{
	const int thread = omp_get_thread_num();
    //AllProbData.clear();
	vector<vector<int> > Ctemp;
	vector<vector<int> > NMinusStemp;
	vector<vector<int> > Cwhole;
		
	Mlgenerate(Ntemp,Ctemp,NMinusStemp,Cwhole, MlT, MulroundVec);
	
	for(int i=0;i<Cwhole.size();i++)
	{
		DynamicTable tempInst1, tempEmpty;
		tempInst1.elm=Cwhole[i];
		MulAllTableElemets.push_back(tempInst1);
		MulDispTableElemets.push_back(tempEmpty);
	}
	
	for(int i=NMinusStemp.size()-1;i>-1;i--)
	{
		DynamicTable tempInst;
		tempInst.elm=NMinusStemp[i];
		MulNSTableElements.push_back(tempInst);
	}
#ifdef _HOST_DEBUG
	cout << "thread: " << thread << ", NSTable size: " << MulNSTableElements.size() << endl;
#endif

    int powK = pow(k,2);

//*****************************************  For optimization for splitting AllTableElements into blocks         ***************************************
//*****************************************  Data displacement should be performed to make block data consective ***************************************
#ifdef SPLIT
//First, Get non-zero maximum configuration, and its size
	//int *maxConfig = new int[powK];
	//int *temp = maxConfig;
	//int count = 0;
	vector<int> maxConfig;
	vector<dim> dimSize;
	vector<int> blockDimSize;
	vector<int> blockDimSizeComp;
	vector<int> divisor;
	vector<int> divisorComp;
	vector<dim> maxN;
	vector<int> inBlockCounterVec;
	vector<block> allBlocks;
	vector<block> allBlocksNoZero;
	vector<int>	blockCounterVec;     			//a vector to store the number of blocks in each level
	block blockVecNoZero(0);
	block blockVec(0);
	block largestBlockVec(0);
	block largestNoZeroBlockVec(0);

	int N = 3;
	int jobsPerBlock = 1;
	int levelsPerBlock = 0;
	int blocksPerConfig = 1;
    int maxJob = 0;
    int maxIndex = MulAllTableElemets.size() - 1;

	for (int i=0; i<MulAllTableElemets[maxIndex].elm.size();i++)
	{
#ifdef _HOST_DEBUG
		cout << "thread: " << thread << ", i: " << i << ", alltableelemets[maxindex].elm[i]: " << MulAllTableElemets[maxIndex].elm[i] << endl;
#endif
		if (MulAllTableElemets[maxIndex].elm[i] != 0)
		{
			maxConfig.push_back(MulAllTableElemets[maxIndex].elm[i]);
		}
	}

//Second, Split AllTableElemets according to the maxConfig.
//Split the multi-dimension table from the highest 3 dimensions. (This value can vary)
//It is better to have a prime number list count from 1 to the max value in maxConfig.

	//increase maxConfig by 1 to get the real size of each dimension.
	dim aa;
	for(int i=0; i<maxConfig.size(); i++)
	{
		aa.weit = maxConfig[i]+1;
		aa.index = i;
		dimSize.push_back(aa);
		maxN.push_back(aa);
		blockDimSize.push_back(maxConfig[i] + 1);
		blockVecNoZero.elm.push_back(0);
#ifdef _HOST_DEBUG
		cout << "thread: " << thread << ", i: "<< i << ", maxN[i].weit: " << maxN[i].weit << ", maxN[i].idx: " << maxN[i].index << ", blockDimSize[i]: " << blockDimSize[i] << endl;
#endif
	}
	

	sort(maxN.begin(), maxN.end(), descSort);
	
	//Store the N largest dimension size into maxN.weight and the index of each max into maxN.index.
	N = min(N, (int)maxConfig.size());
	maxN.resize(N);
//	for(int i =0; i<N; i++)
//		cout << "thread: " << thread << ", i: " << i << ", maxN[i]: " << maxN[i].weit << ", maxN[i].idx: " << maxN[i].index << endl;
	
	
	//For each large dimension size in maxN, find how many segments the dimension should be splitted. If the dimension is not splitted, divisor should be 1.
	for (int i = 0; i < dimSize.size(); i++)
	{
		divisor.push_back(1);
	}

	//this container now may include a dimension's divisor with a size of 1. Future improvement: if it is a prime number, replace the dimension with the next max weight dimension; Or split all dimensions.
	//meanwhile, get the dimension size of each block from divisor and store in blockDimSize
	int div;
	for (int i=0; i<N; i++)
	{
		div = (int)sqrt( (float)maxN[i].weit );
		//cout << "thread: " << thread << ", i: " << i << ", maxN[i]: " << maxN[i].weit << ", div: " << div << endl;
		while (div!=0)
		{
			if (maxN[i].weit % div != 0)
			{
				div--;
				continue;
			}
			//cout << "thread: " << thread << ", i: " << i << ", maxN[i].index: " << maxN[i].index << ", div: " << div << endl;
			if (div == 1){
				divisor[maxN[i].index] = maxN[i].weit;
			}
			else{
				divisor[maxN[i].index] = div;
			}
			blockDimSize[maxN[i].index] /= divisor[maxN[i].index];
			break;
		}
	}

#ifdef _HOST_DEBUG
	for (int i = 0; i < dimSize.size(); i++)
		cout << "thread: " << thread << ", i: " << i << ", divisor[i]: " << divisor[i] << ", blockDimSize[i]: " << blockDimSize[i] << endl;
#endif

	//once get the blockDimSize and divisor, we are able to get jobsPerBlock and blocksPerConfig
	//dimSize.size() is the same as N.
	for (int i=0; i<dimSize.size(); i++)
	{
		jobsPerBlock *= blockDimSize[i];
		levelsPerBlock += blockDimSize[i];
	}

	for (int i = 0; i < divisor.size(); i++)
	{
		blocksPerConfig *= divisor[i];
	}


//	cout << "thread: " << thread << ", jobsPerBlock: " << jobsPerBlock << ", blocksPerConfig: " << blocksPerConfig << endl; 
	
	//It is necessary to get all block vectors and the count of blocks that are in the same level.
	//for (int i = 0; i < blocksPerConfig; i++)
	for (int i = 0; i < powK; i++)
	{
		blockVec.elm.push_back(0);
	}
	blockVec.mySUM = 0;
	allBlocks.push_back(blockVec);

	//expand largestBlockVec, divisor and blockDimSize to the size of powK
	int j = 0;

#ifdef _HOST_DEBUG
	cout << "thread: " << thread << ", largestBlockVec.elm: ";
#endif
	for (int i=0; i<powK;i++)
	{
		if (MulAllTableElemets[maxIndex].elm[i] != 0)
		{
			divisorComp.push_back(divisor[j]);
			blockDimSizeComp.push_back(blockDimSize[j]);
			largestBlockVec.elm.push_back(divisor[j]-1);
			largestNoZeroBlockVec.elm.push_back(divisor[j]-1);
			j++;
		}
		else{
			divisorComp.push_back(0);
			blockDimSizeComp.push_back(0);
			largestBlockVec.elm.push_back(0);
		}
#ifdef _HOST_DEBUG
		cout << largestBlockVec.elm[i] << " ";
#endif
	}
#ifdef _HOST_DEBUG
	cout << endl;

	cout << "thread: " << thread << ", blockDimSizeComp: ";
	for (int i=0; i<powK; i++)
		cout << blockDimSizeComp[i] << " ";
	cout << endl;
	cout << "thread: " << thread << ", divisorComp: ";
	for (int i=0; i<powK; i++)
		cout << divisorComp[i] << " ";
	cout << endl;
	cout << "thread: " << thread << ", largestBlockVec: ";
	for (int i=0; i<powK; i++)
		cout << largestBlockVec.elm[i] << " ";
	cout << endl;
	cout << "thread: " << thread << ", largestNoZeroBlockVec: ";
	for (int i=0; i<largestNoZeroBlockVec.elm.size(); i++)
		cout << largestNoZeroBlockVec.elm[i] << " ";
	cout << endl;

#endif

	while(blockVec.elm != largestBlockVec.elm)
	{
		int index;
		for (int i = 0; i < powK; i++)
		{
			index = powK - 1 - i;
			++blockVec.elm[index];
			++blockVec.mySUM;
			if (blockVec.elm[index] > largestBlockVec.elm[index]) {
				blockVec.mySUM -= blockVec.elm[index];
				blockVec.elm[index] = 0;
			}
			else {
				allBlocks.push_back(blockVec);
				break;
			}
		}
	}
	//sort all block vectors by mySUM. (level)
	sort(allBlocks.begin(), allBlocks.end());

	//get allBlocks Index without 0 value dimensions, maybe not correct and not necessary. May be replaced by inBlockCounterVec.size()
	allBlocksNoZero.push_back(blockVecNoZero);

	while(blockVecNoZero.elm != largestNoZeroBlockVec.elm)
	{
		int index;
		for (int i = 0; i < largestNoZeroBlockVec.elm.size(); i++)
		{
			index = largestNoZeroBlockVec.elm.size() - 1 - i;
			++blockVecNoZero.elm[index];
			++blockVecNoZero.mySUM;
			if (blockVecNoZero.elm[index] > largestNoZeroBlockVec.elm[index]) {
				blockVecNoZero.mySUM -= blockVecNoZero.elm[index];
				blockVecNoZero.elm[index] = 0;
			}
			else {
				allBlocksNoZero.push_back(blockVecNoZero);
				break;
			}
		}
	}
	//sort all block vectors by mySUM. (level)
	sort(allBlocksNoZero.begin(), allBlocksNoZero.end());

#ifdef _HOST_DEBUG
		cout << "thread: " << thread << ", allBlocks: ";
		for (int i = 0; i < allBlocks.size(); i++)
		{
			for (int j=0; j<powK; j++)
				cout << allBlocks[i].elm[j] << " ";
			cout << ", mySum: " << allBlocks[i].mySUM << endl;
		}
#endif

	for (int i = 0; i < allBlocks.size(); i++)
	{
		int c = 1;
		while(i < allBlocks.size() - 1  && allBlocks[i].mySUM == allBlocks[i+1].mySUM)
		{
			c++;
			i++;
		}
		blockCounterVec.push_back(c);
	}
	
//	for (int i=0; i<blockCounterVec.size(); i++)
//		cout << "thread: " << thread << ", i: " << i << ", # of blocks in level " << i << ": " << blockCounterVec[i] << endl;

	// A check may be required on divisor to ensure it is larger than a efficiency rate. If it is lower than the rate, there are too few blocks to be run in parallel.
	//???????????????????????

	//calculate the new position of each configuration and store in MulDispTableElemets
	//MulDispTableElemets.resize( MulAllTableElemets.size() );
	int blockOffset, inBlockOffset, pos;
	vector<int> blockID;
	vector<int> inBlockPos;
	
	
	for (int i = 0; i < MulAllTableElemets.size(); i++)
	{
		//cout << "thread: " << thread << ", i: " << i << ", MulAllTableElemets[i]: ";
		blockID.clear();
		inBlockPos.clear();
		for (int j=0; j<powK; j++)
		{
			if (blockDimSizeComp[j] != 0)
			{
				blockID.push_back( MulAllTableElemets[i].elm[j] / blockDimSizeComp[j] );
				inBlockPos.push_back( MulAllTableElemets[i].elm[j] % blockDimSizeComp[j] );
			}
			//cout << MulAllTableElemets[i].elm[j] << " ";
		}
		//cout << endl;
		//Calculating the offset, which is to find new position according to offset and block ID
		blockOffset = MulDimOffset(blockID, divisor);
		inBlockOffset = MulDimOffset(inBlockPos, blockDimSize);
		pos = blockOffset * jobsPerBlock + inBlockOffset;
		
		//cout << "blockID: ";
		//for (int tt=0; tt<blockID.size(); tt++)
		//{
		//	cout << blockID[tt] << " ";
		//}
		//cout << endl << "blockOffset: " << blockOffset << ", inBlockOffset: " << inBlockOffset << ", pos: " << pos << endl;
		
		MulDispTableElemets[pos] = MulAllTableElemets[i];
		//std::copy(&MulAllTableElemets[i], &MulAllTableElemets[i+1], &MulDispTableElemets[pos]);
	}
/*	
	for (int i=0; i<MulDispTableElemets.size(); i++)
	{
		cout << "thread: " << thread << ", i: " << i << ", MulDispTableElemets[i]: ";
		for (int j=0; j<powK; j++)
		{
			cout << MulDispTableElemets[i].elm[j] << " ";
		}
		cout << endl;
	}
*/

//Third: Sort each block and count how many configurations per level within each box
	//Parallelism inside of each block is the same as AllTableElemets for configurations that are at same level get run in parallel. Thus, configurations in each block can be sorted within block.
	for (int i=0; i<MulDispTableElemets.size(); i++) 	//Calculate the # of configurations in each level for a splitted block
	{
		int sumOver=0;
		for (int j=0; j<MulDispTableElemets[i].elm.size(); j++)   //AllTableElemets[i].elm.size() = Cwhole[i].szie() = Ntemp.size()
		{
			sumOver  = sumOver+ MulDispTableElemets[i].elm[j];
		}
		//inBlockLevelSum.push_back(sumOver);
		MulDispTableElemets[i].mySum=sumOver;
		//cout << "thread: " << thread << ", i: " << i << ", MulDispTableElemets[i].mySum: " << MulDispTableElemets[i].mySum << endl;
	}

	//sort(inBlockLevelSum.begin(), inBlockLevelSum.end());			//align jobs in order for dependency.

	//this might not work, if not, try use MulAllTableElemets.data().
	for (vector<DynamicTable>::iterator it = MulDispTableElemets.begin(); it != MulDispTableElemets.end(); it+=jobsPerBlock)
	{
		sort(it, it+jobsPerBlock);
	}
#ifdef _HOST_DEBUG
/*	
	for (int block = 0; block < allBlocks.size(); block++)
	{
		cout << "thread: " << thread << ", block: " << block << ", vectors: " << endl;
		int offset = block * jobsPerBlock;
		for (int i = 0; i < jobsPerBlock; i++)
		{
			cout << "vector " << i << ": ";
			int offset1 = offset + i;
			for (int j = 0; j<powK; j++)
				cout << MulDispTableElemets[offset1].elm[j] << " ";
			cout << endl;
		}
	}
*/
#endif
	// count how many of each item is in the inBlockLevelSum
	for(int i = 0; i < jobsPerBlock; i++)		//the number of jobs in each level
	{
		int c = 1;
		int limit = jobsPerBlock - 1;
		while(i < limit  && MulDispTableElemets[i].mySum == MulDispTableElemets[i+1].mySum)
		{
			i++;
			c++;
		}
		//cout << "thread: " << thread << ", i: " << i << ", c: " << c << endl;
		inBlockCounterVec.push_back(c);
	}

	//maxSumValue = number of total jobs + 1
	gpu_BlockDP(MulDispTableElemets, MlT, powK, jobsPerBlock, levelsPerBlock, inBlockCounterVec,
	    		MulLongJobs.size(), &zeroVec[0], &MulroundVec[0], divisor, divisorComp,
	    		blockDimSizeComp, allBlocks, allBlocksNoZero, blockCounterVec);

#ifdef _HOST_DEBUG
	cout << "Thread: " << thread << ", gpu_BlockDP is done." << endl;
#endif

	for(int i=0; i<MulNSTableElements.size();i++)			//NSTableElements is N - S. For example. (2,3), si = (0,1), then NS[i] = (2,2)
	{
		for(int j=0;j<MulDispTableElemets.size();j++)
		{
			if(MulNSTableElements[i].elm==MulDispTableElemets[j].elm)
			{
//				cout << "thread: " << thread << ", i: " << i << ", NSTable[i]: ";
				MulNSTableElements[i]=MulDispTableElemets[j];

//				for (int tt = 0; tt < powK; tt++)
//					cout << MulNSTableElements[i].elm[tt] << " ";
//				cout << ", NSTable.myOPT: " << MulNSTableElements[i].myOPT << ", DispTable.myOPT: " << MulDispTableElemets[j].myOPT;
//				cout << endl;
				break;
			}
		}

	}

#ifdef _HOST_DEBUG
	cout << "Thread: " << thread << ", find the NS table for a selected configuration." << endl;
#endif

	for(int i=0; i<MulNSTableElements.size();i++)
	{
		MultempOptVector.push_back(MulNSTableElements[i].myOPT);
	}

	int dpoptimal;
	int minN=100000;

//	cout << "NSTableElemets size: " << MulNSTableElements.size() << endl;
	for(int mindex=0; mindex<MulNSTableElements.size();mindex++)
	{
		if(MulNSTableElements[mindex].myOPT<minN)
			minN=MulNSTableElements[mindex].myOPT;
	}

	dpoptimal=minN +1;
	
#ifdef _HOST_DEBUG
	cout << "cpuId: " << thread << ", dpoptimal: " <<dpoptimal << endl;
#endif

	clock_t ttt = clock();

	if (dpoptimal<=nMachines)		//keep a copy of the latest feasible solution.
	{

		FinalTableINFO instance;
		instance.AllTableElemets=MulDispTableElemets;
		instance.NSTableElements=MulNSTableElements;
		instance.Nvector=MulNtemp;
		instance.OPTtable=dpoptimal;
		instance.Ttable= MlT;
		instance.optimalValuesVector=MultempOptVector;
		instance.ShortJobsTable=MulShortJobs;
		instance.LongJobsTable=MulLongJobs;
		instance.roundVecTable=MulroundVec;
		instance.LongRoundJobsTable=MulLongRoundJobs;
		instance.roundCriteriaTable=roundCriteria;
#pragma omp critical
{
		MulAllProbData.push_back(instance);
}
	}

	ttt = clock() - ttt;
	
#ifdef _HOST_DEBUG
	cout << "copy the latest feasible solution to instance takes time: " << (float)ttt/CLOCKS_PER_SEC << endl;
#endif
//*****************************************  Here is the end of AllTableElemets split work *************************************************************
#else

    vector<int> sumVec;
	for (int i=0; i<MulAllTableElemets.size(); i++) 	//calculate the total run time of each job, jobs have same run time are in the same level (can do parallel).  AllTableElemets.size() = CWhole.size()
	{
		int sumOver=0;
		for (int j=0; j<MulAllTableElemets[i].elm.size(); j++)   //AllTableElemets[i].elm.size() = Cwhole[i].szie() = Ntemp.size()
		{
			sumOver  = sumOver+ MulAllTableElemets[i].elm[j];
		}
		sumVec.push_back(sumOver);
        MulAllTableElemets[i].mySum=sumOver;
	}
    
    sort(sumVec.begin(), sumVec.end());			//align jobs in order for dependency.
    vector<int> counterVec;
    
    // count how many of each item is in the sumVec
    for(int i = 0; i < sumVec.size(); i++)		//the number of jobs in each level       sumVec.size() = AllTableElemets.size()
    {
        int c = 1;
        
        int limit = sumVec.size() - 1;
        while(i < limit  && sumVec[i] == sumVec[i+1])
        {
            c++;
            i++;
        }
        
        counterVec.push_back(c);
    }
    
    sort(MulAllTableElemets.begin(), MulAllTableElemets.end());

	int maxSumIndex;
	maxSumIndex = max_element(sumVec.begin(), sumVec.end()) - sumVec.begin();			
	int maxSumValue = sumVec[maxSumIndex];											//the max time among all configuration

	vector<int> zeroVec;
	for(int i=0;i< Ntemp.size();i++)
		zeroVec.push_back(0);

    if (powK > 1024)
    {
    	cout << "GPU function does not work because powK is over 1024." << endl;
    	exit(-1);
    }

    gpu_DP(MulAllTableElemets, MlT, k, powK, maxSumValue, counterVec,
    		MulLongJobs.size(), &zeroVec[0], &MulroundVec[0]);

    for(int i=0; i<MulNSTableElements.size();i++)			//NSTableElements is N - S. For example. (2,3), si = (0,1), then NS[i] = (2,2)
    	{
    		for(int j=0;j<MulAllTableElemets.size();j++)
    		{
    			if(MulNSTableElements[i].elm==MulAllTableElemets[j].elm)
    			{
    				MulNSTableElements[i]=MulAllTableElemets[j];
    				break;
    			}
    		}

    	}

    	for(int i=0; i<MulNSTableElements.size();i++)
    	{
    		MultempOptVector.push_back(MulNSTableElements[i].myOPT);
    	}

    	int dpoptimal;
    	int minN=100000;

    //	cout << "NSTableElemets size: " << MulNSTableElements.size() << endl;
    	for(int mindex=0; mindex<MulNSTableElements.size();mindex++)
    	{
    		if(MulNSTableElements[mindex].myOPT<minN)
    			minN=MulNSTableElements[mindex].myOPT;
    	}

    	dpoptimal=minN +1;

        cout << "dpoptimal: " <<dpoptimal << endl;
        
        clock_t ttt = clock();

        if (dpoptimal<=nMachines)		//keep a copy of the latest feasible solution.
        {

            FinalTableINFO instance;
            instance.AllTableElemets=MulAllTableElemets;
            instance.NSTableElements=MulNSTableElements;
            instance.Nvector=MulNtemp;
            instance.OPTtable=dpoptimal;
            instance.Ttable= MlT;
            instance.optimalValuesVector=MultempOptVector;
            instance.ShortJobsTable=MulShortJobs;
            instance.LongJobsTable=MulLongJobs;
            instance.roundVecTable=MulroundVec;
            instance.LongRoundJobsTable=MulLongRoundJobs;
            instance.roundCriteriaTable=roundCriteria;
#pragma omp critical
{
            MulAllProbData.push_back(instance);
}
        }

    	ttt = clock() - ttt;
    	cout << "copy the latest feasible solution to instance takes time: " << (float)ttt/CLOCKS_PER_SEC << endl;

#endif


#ifdef SPLIT
	maxConfig.clear();
	dimSize.clear();
	blockDimSize.clear();
	divisor.clear();
	maxN.clear();
	inBlockCounterVec.clear();
	Cwhole.clear();
	Ctemp.clear();
	NMinusStemp.clear();
#else
	sumVec.clear();
	Cwhole.clear();
	Ctemp.clear();
	NMinusStemp.clear();
#endif
	return dpoptimal;
}


void generate(vector<int>& Ntemp, vector<vector<int> >& Ctemp, vector<vector<int> >& NMinusStemp,vector<vector<int> >& Cwhole)
{
  vector<int> it(Ntemp.size(), 0);   //"it" is the vector that there is one new long job assigned to in each iteration, in the descending order on round events' time.
  do {    //this loop iterates through all possible subsets that are no larger than "Ntemp".
		vector<int> s;
		for(vector<int>::const_iterator i = it.begin(); i != it.end(); ++i)
		{
			s.push_back(*i);
		}
		Cwhole.push_back(s);             //"s" is the same to it, "s" points out the new long event assigned in current iteration;
		                                 //"Cwhole" has all machine configurations. "s" is one configuration
		int sSum=sumFun(s,roundVec);     //roundVec is the vector of the run time of all round events, sumFun(s, roundVec) is to get the current sum of all assigned long events.
		if(sSum <= T)
		{
			Ctemp.push_back(s);
			vector<int> NS;          //NS is the updated vector that has unassigned long works for round down timings.
			for(int j=0; j<Pow(k,2); j++)
			{
			  NS.push_back( Ntemp[j]-s[j]);    //Ntemp is previously created, which is the OPT(configuration). Ntemp - s is one of the next OPT(configuration).
			}
			if(NS==Ntemp)            //if there is no long work assigned in this iteration, go to next iteration.
				continue;
			NMinusStemp.push_back(NS); //set of vectors for remaining long works. Updated by iteration.
		}
	}while (increase(Ntemp, it));
}

void Mlgenerate(vector<int>& Ntemp, vector<vector<int> >& Ctemp, vector<vector<int> >& NMinusStemp,vector<vector<int> >& Cwhole, const int MlT,
				vector<int>& MulroundVec)
{		
	const int thread = omp_get_thread_num();
	
	vector<int> it(Ntemp.size(), 0);   //"it" is the vector that there is one new long job assigned to in each iteration, in the descending order on round events' time.
	do {    //this loop iterates through all possible subsets that are no larger than "Ntemp".
		vector<int> s;
		for(vector<int>::const_iterator i = it.begin(); i != it.end(); ++i)
		{
			s.push_back(*i);
		}
		Cwhole.push_back(s);             //"s" is the same to it, "s" points out the new long event assigned in current iteration;
										 //"Cwhole" has all machine configurations. "s" is one configuration		
		int sSum=sumFun(s, MulroundVec);     //roundVec is the vector of the run time of all round events, sumFun(s, roundVec) is to get the current sum of all assigned long events.
		
		if(sSum <= MlT)
		{
			Ctemp.push_back(s);
			vector<int> NS;          //NS is the updated vector that has unassigned long works for round down timings.
			for(int j=0; j<Pow(k,2); j++)
			{
			  NS.push_back( Ntemp[j]-s[j]);    //Ntemp is previously created, which is the OPT(configuration). Ntemp - s is one of the next OPT(configuration).
			}
			if(NS==Ntemp)            //if there is no long work assigned in this iteration, go to next iteration.
				continue;
			NMinusStemp.push_back(NS); //set of vectors for remaining long works. Updated by iteration.
		}
	}while (increase(Ntemp, it));
}

void generate2(vector<int>& Ntemp, vector<vector<int> >& Ctemp, vector<vector<int> >& NMinusStemp)
{
    vector<int> it(Ntemp.size(), 0);
    do {
        vector<int> s;
        for(vector<int>::const_iterator i = it.begin(); i != it.end(); ++i)
        {
            s.push_back(*i);
        }
        //Cwhole.push_back(s);
        int sSum=sumFun(s,roundVec);
        if(sSum <= T)
        {
               Ctemp.push_back(s);
            
            vector<int> NS;
            for(int j=0; j<Pow(k,2); j++)
            {
                NS.push_back( Ntemp[j]-s[j]);
            }
            if(NS==Ntemp)
                continue;
            NMinusStemp.push_back(NS);
        }
    }while (increase(Ntemp, it));
}


void printFinalSchedule(vector<vector<int> >& optimalSchedule,vector<int>& Machinetimes, vector<int>& longF,vector<int>& shortF, int Fopt)
{
   /* scheduleFile << "Fopt"<<"\t"<<Fopt<<endl;
    scheduleFile << "Long Jobs:"<<endl;
    for (int i=0; i < longF.size(); i++)
    {
        scheduleFile << longF[i] << "\t";
    }
    scheduleFile << endl;
    scheduleFile << "Short Jobs:"<<endl;
    for (int i=0; i<shortF.size(); i++)
    {
        scheduleFile << shortF[i] <<"\t";
    }
    scheduleFile << endl;*/

/*    scheduleFile << "final schedule"<<endl;
    for (int i=0; i<nMachines; i++)
    {
        scheduleFile << "M"<<i+1<<"\t";
        for (int j=0; j<optimalSchedule[i].size(); j++) {
            scheduleFile << optimalSchedule[i][j] << "\t";
        }
        scheduleFile << endl;
    }
    
    scheduleFile << "Machine times"<<endl;
    for (int i=0; i<nMachines; i++)
    {
        scheduleFile << "M"<<i+1<<"\t"<< Machinetimes[i]<<endl;
    }
    scheduleFile << "Makespan"<<"\t"<<FinalMakespan<<endl;
  */  
}


void findScheduleFun3(vector<vector<int> >& FinalMachineConfiguration, vector< vector <int> >& RoundedOptimalSchedule)
{
    vector < vector < vector <int> > > schedulingTable;
    vector < vector< int> > temp2;
    vector<int> temp1;
    
    for (int ck=0; ck < Pow(k,2); ck++)
    {
        temp1.push_back(0);
    }
    for (int ck=0; ck < 2; ck++)
    {
        temp2.push_back(temp1);
    }
    for (int ck=0; ck < nMachines; ck++)
    {
        schedulingTable.push_back(temp2);
    }
    
    int gh=AllProbData[optIndex].AllTableElemets.size();
   // cout << "---------------------------findScheduleFun 3--------------------------- "<< endl;
//    cout<< "gh" << gh<<endl;
    
    schedulingTable[0][0]= AllProbData[optIndex].AllTableElemets[gh-1].elm;
    schedulingTable[0][1]= AllProbData[optIndex].AllTableElemets[gh-1].myMinNSVector;
    
    for (int gf=1; gf<nMachines; gf++)
    {

        for (int ck=0; ck<gh; ck++)
        {
            if (AllProbData[optIndex].AllTableElemets[ck].elm == schedulingTable[gf-1][1])
            {
                schedulingTable[gf][0]=AllProbData[optIndex].AllTableElemets[ck].elm;
                schedulingTable[gf][1]= AllProbData[optIndex].AllTableElemets[ck].myMinNSVector;
                break;
            }
        }
        if(schedulingTable[gf][1] == zeroVec)
        {
        //    cout << "gf last "<<gf<<endl;
            break;
        }

    }
   // cout << "schedulingTable" <<endl;
  //  cout<<" {"<<endl;
  //  for (int q=0; q<schedulingTable.size(); q++) {
   //      cout<<" {";
  //      for (int w=0; w<schedulingTable[q].size(); w++) {
   //         cout<<" {";
    //        for (int e=0; e<schedulingTable[q][w].size(); e++) {
    //            cout<<schedulingTable[q][w][e]<< "  ";
    //        }
    //        cout<<" },";
    //    }
   //     cout<<"},"<<endl;
  //  }
  //  cout<<" }"<<endl;
    vector < vector <int> > MyRealSchedule;
    vector < vector <int> > MySchedule;
    vector<int> tempEmptyVec;
    for (int ck=0; ck < nMachines; ck++)
    {
        MySchedule.push_back(temp1);
        MyRealSchedule.push_back(tempEmptyVec);
    }
    
    for (int q=0; q<schedulingTable.size(); q++)
    {
            for (int e=0; e<schedulingTable[q][0].size(); e++)
            {
                temp1[e]=schedulingTable[q][0][e]-schedulingTable[q][1][e];
            }
        MySchedule[q]=temp1;
    }
    
   // cout << "My new schedule" << endl;
   // for (int q=0; q<schedulingTable.size(); q++)
  //  {
   //     for (int e=0; e<schedulingTable[q][0].size(); e++)
   //     {
   //         cout << MySchedule[q][e] << " ";
   //     }
   //     cout<< endl;
   // }
    
    
    
    
    for(int i=0; i<MySchedule.size();i++)
    {
        for(int j=0;j<MySchedule[i].size();j++)
        {
            if(MySchedule[i][j]!=0)
            {
                for(int d=0;d<MySchedule[i][j];d++)
                {
                    MyRealSchedule[i].push_back(AllProbData[optIndex].roundVecTable[j]); //*****
                }
            }
        }
    }
    
    
   // cout << "My new REAL schedule" << endl;
  //  for (int q=0; q<MyRealSchedule.size(); q++)
  //  {
  //      for (int e=0; e<MyRealSchedule[q].size(); e++)
     //   {
    //        cout << MyRealSchedule[q][e] << " ";
    //    }
    //    cout<< endl;
   // }
    
    FinalMachineConfiguration=MySchedule;
    RoundedOptimalSchedule=MyRealSchedule;
    
}

void findScheduleFun2(vector<vector<int> >& FinalMachineConfiguration, vector< vector <int> >& RoundedOptimalSchedule)
{
	vector <int> temp;
	int ii,id,index;
	//optIndex=AllProbData.size()-1;
    cout << "---------------------------findScheduleFun2--------------------------- "<< endl;
   // cout << "AllProbData.size()     " << AllProbData.size()<< endl;
   // cout << "optIndex    "<< optIndex<<endl;
   // cout << "AllProbData[optIndex].NSTableElements.size()       "<< AllProbData[optIndex].NSTableElements.size()<<endl;
   // cout << " AllProbData[optIndex].optimalValuesVector" << endl;
   // for (int ck=0; ck<AllProbData[optIndex].optimalValuesVector.size(); ck++) {
    //    cout << AllProbData[optIndex].optimalValuesVector[ck]<< " ";
    //}
    //cout << endl;
    
    
    // Here we are trying to find the schedule
    // To do so we should find out how we found the optimal number of machine, basically we found it from the formula   ..... > OPT = 1+ min OPT (NS)
    // So that in the first step we should find NS that made our minimun in the formula , using following expression :
    // We try to find its index first
    
    int gh=AllProbData[optIndex].AllTableElemets.size();
    //cout << "check for ii   "<<AllProbData[optIndex].AllTableElemets[gh-1].myOptimalindex<<endl;
    //cout << " LAst element of AllProbData[optIndex].AllTableElemets " << endl;
    //for (int ck=0; ck<AllProbData[optIndex].AllTableElemets[gh-1].elm.size(); ck++) {
     //   cout << AllProbData[optIndex].AllTableElemets[gh-1].elm[ck]<< " ";
   // }
    //cout<< endl;
    
    
    //cout << "AllProbData[optIndex].AllTableElemets[gh-1].optVector" << endl;
    //for (int ck=0; ck<AllProbData[optIndex].AllTableElemets[gh-1].optVector.size(); ck++) {
     //   cout << AllProbData[optIndex].AllTableElemets[gh-1].optVector[ck]<< " ";
    //}
    //cout << endl;
    
    
	ii = min_element(AllProbData[optIndex].optimalValuesVector.begin(),AllProbData[optIndex].optimalValuesVector.end()) - AllProbData[optIndex].optimalValuesVector.begin();
    
    // Now we found its index which we call it  ( ii )
    //cout << "ii     "<<ii<<endl;
	temp=AllProbData[optIndex].Nvector;
    
    // Printig to check .....
    
    cout << " AllProbData[optIndex].Nvector" << endl;
    for (int ck=0; ck<AllProbData[optIndex].Nvector.size(); ck++) {
        cout << AllProbData[optIndex].Nvector[ck]<< " ";
    }
    cout << endl;
    
    // Now that we found this minimum came from where , so we should found the configuration corresponding to it that shows the first machine schedule
    // we call it temp and then we will add it to our schedule  .... > FinalMachineConfiguration
    
	for(int i=0;i<AllProbData[optIndex].Nvector.size();i++)
	{
		temp[i]=AllProbData[optIndex].Nvector[i]- AllProbData[optIndex].NSTableElements[ii].elm[i];
	}
    

    cout << " AllProbData[optIndex].NSTableElements[ii].elm" << endl;
    for (int ck=0; ck<AllProbData[optIndex].NSTableElements[ii].elm.size(); ck++) {
        cout << AllProbData[optIndex].NSTableElements[ii].elm[ck]<< " ";
    }
    cout << endl;
    cout << endl;

    
    
    cout << " Temp " << endl;
    for (int ck=0; ck<temp.size(); ck++) {
        cout << temp[ck]<< " ";
    }
    cout << endl;

    
    // Here we will push our first machine schedule to the FinalMachineConfiguration
    
	FinalMachineConfiguration.push_back(temp);

    
    // Now, we should find the schedules of the remaining machines
    
	for(int i=0;i<AllTableElemets.size();i++)
	{
		if(AllProbData[optIndex].AllTableElemets[i].elm ==AllProbData[optIndex].NSTableElements[ii].elm)
		{
			index=i;
			break;
		}	
	}

    //cout << "index     "<<index<<endl;

    
    
    
    
    
	do{
        
        
      //  cout << " AllProbData[optIndex].AllTableElemets[index].optVector " << endl;
       // for (int ck=0; ck<AllProbData[optIndex].AllTableElemets[index].optVector.size(); ck++) {
        //    cout << AllProbData[optIndex].AllTableElemets[index].optVector[ck]<< " ";
       // }
       // cout << endl;
        
        
        
		id=min_element(AllProbData[optIndex].AllTableElemets[index].optVector.begin(),AllProbData[optIndex].AllTableElemets[index].optVector.end())-AllProbData[optIndex].AllTableElemets[index].optVector.begin();
		//id = min_element(AllTableElemets[index].optVector.begin(),AllTableElemets[index].optVector.end()) - AllTableElemets[index].optVector.begin();
		//AllProbData[optIndex].AllTableElemets[index].NSsubsets[id];
        
        //cout << "id   " << id << endl;
        //cout <<"AllProbData[optIndex].AllTableElemets[index].Csubsets.size      "<< AllProbData[optIndex].AllTableElemets[index].Csubsets.size()<<endl;
        //cout <<"AllProbData[optIndex].AllTableElemets[index].NSsubsets.size     "<< AllProbData[optIndex].AllTableElemets[index].NSsubsets.size()<<endl;
        //cout << "============================================================================="<<endl;
        
        //***********************
        //cout << " NEXT of FinalMachineConfiguration " << endl;
        //for (int ck=0; ck<AllProbData[optIndex].AllTableElemets[index].Csubsets[id+1].size(); ck++) {
         //   cout << AllProbData[optIndex].AllTableElemets[index].Csubsets[id+1][ck]<< " ";
       // }
        //cout << endl;
        
		FinalMachineConfiguration.push_back(AllProbData[optIndex].AllTableElemets[index].Csubsets[id+1]);

		if(AllProbData[optIndex].AllTableElemets[index].myOPT==1)
			break;

		for(int i=0;i<AllTableElemets.size();i++)
		{
			if(AllProbData[optIndex].AllTableElemets[i].elm==AllProbData[optIndex].AllTableElemets[index].NSsubsets[id])
			{
				index=i;
				//FinalMachineConfiguration.push_back(AllTableElemets[i].elm);
				break;
			}	
		}



	}while(1);

    cout <<" FinalMachineConfiguration.size() " << FinalMachineConfiguration.size()<<endl;
    cout <<" RoundedOptimalSchedule.size() " << RoundedOptimalSchedule.size()<<endl;
    
	for(int i=0; i<FinalMachineConfiguration.size();i++)
	{
		for(int j=0;j<FinalMachineConfiguration[i].size();j++)
		{
			if(FinalMachineConfiguration[i][j]!=0)
			{
				for(int d=0;d<FinalMachineConfiguration[i][j];d++)
				{
					RoundedOptimalSchedule[i].push_back(AllProbData[optIndex].roundVecTable[j]); //*****
				}
			}
		}
	}
    
    
    cout << "FinalMachineConfiguration" << endl;
    for (int q=0; q<FinalMachineConfiguration.size(); q++)
    {
        for (int e=0; e<FinalMachineConfiguration[q].size(); e++)
        {
            cout << FinalMachineConfiguration[q][e] << " ";
        }
        cout<< endl;
    }
    
    
    
    cout << "RoundedOptimalSchedule" << endl;
    for (int q=0; q<RoundedOptimalSchedule.size(); q++)
    {
        for (int e=0; e<RoundedOptimalSchedule[q].size(); e++)
        {
            cout << RoundedOptimalSchedule[q][e] << " ";
        }
        cout<< endl;
    }
    
    
}

void printFunFile(vector<int> v)
{
	for(int j=0; j<v.size(); j++)
	{
//		resultFile3 << v[j]<< "\t ";
	}
//	resultFile3<<"\n"<<endl;
}


void printFun(vector<int> v)
{
	for(int j=0; j<v.size(); j++)
	{
 //       outputFile << v[j]<< ' ';
		cout<< v[j]<< ' ';
	}
//    outputFile <<"\n"<<endl;
	cout<<"\n"<<endl;
}

void printFun(vector<double> v)
{
	for(int j=0; j<v.size(); j++)
	{
		cout<< v[j]<< ' ';
//        outputFile << v[j]<< ' ';
	}
	cout<<"\n"<<endl;
//     outputFile <<"\n"<<endl;
}

void clearFun()
{

	Ntemp.clear();
	LongRoundJobs.clear();
	LongJobs.clear();
	ShortJobs.clear();
	roundVec.clear();

}

void MlclearFun(const int seg,  vector <vector<FinalTableINFO> >& MulAllProbData)
{
	for (int x = 0; x < seg; x++)
		{
			MulAllProbData[x].clear();
		}
}


void ListSchedulingFun(vector<int>& ShortJobs,vector<vector<int> >& OptimalSchedule,vector<int>& MachtimeI,int m)
{
	//vector<int> SortedJobs;
	//SortedJobs=ShortJobs;
	int nn=ShortJobs.size();
	sort (ShortJobs.rbegin(), ShortJobs.rbegin()+nn);

	int index;
	for(int i=0;i<nn;i++)
	{
		index=min_element(MachtimeI.begin(), MachtimeI.end()) - MachtimeI.begin();
		OptimalSchedule[index].push_back(ShortJobs[i]);
		MachtimeI[index]=MachtimeI[index]+ShortJobs[i];
	}
}

bool increase(const vector<int>& Ntemp, vector<int>& it)
{
	for (int i = 0, size = it.size(); i != size; ++i) {
		const int index = size - 1 - i;
		++it[index];
		if (it[index] > Ntemp[index]) {
			it[index] = 0;
		} else {
			return true;
		}
	}
	return false;
}

int Pow(int x, int p)
{
	if (p == 0) return 1;
	if (p == 1) return x;

	int tmp = Pow(x, p/2);
	if (p%2 == 0) return tmp * tmp;
	else return x * tmp * tmp;
}

int sumFun(vector<int> A, vector<int> B)
{
	int summ=0.0;
	for(int i=0; i<(Pow(k,2)); i++)
	{
		summ= summ + A[i]*B[i];
	}
	return summ;
}

int rounDownFun(vector<int>& roundVec, vector<int> L,int iterator)
{

	for(int i=iterator; i<L.size();i++)
	{
		for(int j=0;j<roundVec.size();j++)
		{
			if(roundVec[j] <= L[i])
			{
				if(j != roundVec.size() -1 )
				{
					if(roundVec[j+1] > L[i])
						return roundVec[j];
				}
				else if(j == roundVec.size() -1 )
				{
					return roundVec[j];
				}
			}


		}
	}

}

void print(int iwhile)
{
    
    
  /*  outputFile <<"**********************    B_k   Algorithm   ************************************"<<endl;
    outputFile<<"Iterate             :  "<< iwhile <<endl;
    outputFile<<"'Error' is equal to :  "<< error <<endl;
    outputFile<<"'k' is  equal to    :  "<< k <<endl;
    outputFile<<"'Upper Bound' is    :  "<< UB <<endl;
    outputFile<<"'Lower Bound' is    :  "<< LB <<endl;
    outputFile<<"'T' is  equal to    :  "<< T <<endl;
    outputFile<<"Short Jobs : "<< endl;
   */
	cout<<"**********************    B_k   Algorithm   ************************************"<<endl;
	cout<<"Iterate             :  "<< iwhile <<endl;
	cout<<"'Error' is equal to :  "<< error <<endl;
	cout<<"'k' is  equal to    :  "<< k <<endl;
	cout<<"'Upper Bound' is    :  "<< UB <<endl;
	cout<<"'Lower Bound' is    :  "<< LB <<endl;
	cout<<"'T' is  equal to    :  "<< T <<endl;
	cout<<"Short Jobs : "<< endl;
    
	printFun(ShortJobs);
	cout<<"Long Jobs : "<< endl;
  //  outputFile <<"Long Jobs : "<< endl;
	printFun(LongJobs);
	cout<<" T/K^2 vector : "<< endl;
   // outputFile <<" T/K^2 vector : "<< endl;
	printFun(roundVec);
	cout<<"Rounded down Long Jobs "<< endl;
   // outputFile <<"Rounded down Long Jobs "<< endl;
	printFun(LongRoundJobs);
	cout<<"The vector of number of long jobs of rounded size equal to iT/K^2 "<< endl;
   // outputFile <<"The vector of number of long jobs of rounded size equal to iT/K^2 "<< endl;
	printFun(Ntemp);
	cout<<"The minimum number of machines sufficient to schedule the above input  is     "<< OPT<<endl;
    //outputFile<<"The minimum number of machines sufficient to schedule the above input  is     "<< OPT<<endl;

/*	resultFile3<<"Iterate:"<< "\t" << iwhile <<endl;
	resultFile3<<"'Error' is equal to:"<< "\t" << error <<endl;
	resultFile3<<"'k' is  equal to:"<< "\t" << k <<endl;
	resultFile3<<"'Upper Bound is:"<< "\t" << UB <<endl;
	resultFile3<<"'Lower Bound is:"<< "\t" << LB <<endl;
	resultFile3<<"'T' is  equal to:"<< "\t" << T <<endl;
	resultFile3<<"Short Jobs : "<< endl;*/
	printFunFile(ShortJobs);
//	resultFile3<<"Long Jobs : "<< endl;
	printFunFile(LongJobs);
//	resultFile3<<" T/K^2 vector : "<< endl;
	printFunFile(roundVec);
//	resultFile3<<"Rounded down Long Jobs "<< endl;
	printFunFile(LongRoundJobs);
//	resultFile3<<"The vector of number of long jobs of rounded size equal to iT/K^2 "<< endl;
	printFunFile(Ntemp);
//	resultFile3<<"DP OPT"<< "\t" <<  OPT <<endl;

	cout<<"********************************************************************************"<<endl;
}


void readInputFile(int nFile)
{
	//reading from the file :
   // outputFile <<"Reading File..."<<endl;
	cout<<"Reading File..."<<endl;
	ifstream myfile;

    if(nFile==10)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File10-worst-J21-M10-R10-19.txt");
    else if(nFile==11)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File11-semi-worst-J30-M10-R1-19.txt");
    else if(nFile==12)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File12-semi-worst-J30-M10-R5-19.txt");
    else if(nFile==13)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File13-semi-worst-J38-M10-R1-19.txt");
    
    
    else if(nFile==15)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File15-Pareto-J30-M10-R1-19-a0.5.txt");
    else if(nFile==16)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File16-Pareto-J30-M10-R1-19-a0.63.txt");
    else if(nFile==17)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File17-Pareto-J30-M10-R1-19-a0.9.txt");
    else if(nFile==18)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File18-Pareto-J30-M10-R1-19-a1.3.txt");
    
    
    else if(nFile==20)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File20-worst-J41-M20-R20-39.txt");
    
    else if(nFile==21)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File21-semi-worst-J60-M20-R1-39.txt");
    else if(nFile==22)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File22-semi-worst-J60-M20-R10-39.txt");
    else if(nFile==23)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File23-semi-worst-J78-M20-R1-39.txt");
    
    else if(nFile==25)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File25-Pareto-J60-M20-R1-39-a0.5.txt");
    else if(nFile==26)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File26-Pareto-J60-M20-R1-39-a0.63.txt");
    else if(nFile==27)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File27-Pareto-J60-M20-R1-39-a0.9.txt");
    else if(nFile==28)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File28-Pareto-J60-M20-R1-39-a1.3.txt");
    
    
    
    else if(nFile==30)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File30-worst-J61-M30-R30-59.txt");
    else if(nFile==31)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File31-semi-worst-J90-M30-R1-59.txt");
    
    
    else if(nFile==35)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File35-Pareto-J90-M30-R1-59-a0.5.txt");
    else if(nFile==36)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File36-Pareto-J90-M30-R1-59-a0.63.txt");
    else if(nFile==37)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File37-Pareto-J90-M30-R1-59-a0.9.txt");
    else if(nFile==38)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File38-Pareto-J90-M30-R1-59-a1.3.txt");
    
    
    else if(nFile==40)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File40-worst-J81-M40-R40-79.txt");
    else if(nFile==41)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File41-semi-worst-J120-M40-R1-79.txt");
    
    
    
    else if(nFile==45)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File45-Pareto-J120-M40-R1-79-a0.5.txt");
    else if(nFile==46)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File46-Pareto-J120-M40-R1-79-a0.63.txt");
    else if(nFile==47)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File47-Pareto-J120-M40-R1-79-a0.9.txt");
    else if(nFile==48)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File48-Pareto-J120-M40-R1-79-a1.3.txt");
    
    
    
    
    else if(nFile==50)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File50-worst-J101-M50-R50-99.txt");
    else if(nFile==51)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File51-semi-worst-J150-M50-R1-99.txt");
    
    
    
    
    
    else if(nFile==55)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File55-Pareto-J150-M50-R1-99-a0.5.txt");
    else if(nFile==56)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File56-Pareto-J150-M50-R1-99-a0.63.txt");
    else if(nFile==57)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File57-Pareto-J150-M50-R1-99-a0.9.txt");
    else if(nFile==58)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File58-Pareto-J150-M50-R1-99-a1.3.txt");
    else if(nFile==59)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File59-Pareto-J150-M50-R1-99-a1.2.txt");
    
    
    
    
    else if(nFile==60)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File60-worst-J121-M60-R60-119.txt");
    else if(nFile==61)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File61-semi-worst-J180-M60-R1-119.txt");
    
    
    
    
    else if(nFile==65)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File65-Pareto-J180-M60-R1-119-a0.5.txt");
    else if(nFile==66)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File66-Pareto-J180-M60-R1-119-a0.63.txt");
    else if(nFile==67)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File67-Pareto-J180-M60-R1-119-a0.9.txt");
    else if(nFile==68)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File68-Pareto-J180-M60-R1-119-a1.3.txt");
    else if(nFile==69)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File69-Pareto-J180-M60-R1-119-a1.2.txt");
    
    else if(nFile==70)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File70-worst-J141-M70-R70-139.txt");
    else if(nFile==71)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File71-semi-worst-J210-M70-R1-139.txt");
    else if(nFile==78)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File78-Pareto-J210-M70-R1-139-a1.3.txt");
    else if(nFile==79)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File79-Pareto-J210-M70-R1-139-a1.2.txt");
    
    
    else if(nFile==80)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File80-worst-J161-M80-R80-159.txt");
    else if(nFile==81)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File81-semi-worst-J240-M80-R1-159.txt");
    else if(nFile==88)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File88-Pareto-J240-M80-R1-159-a1.3.txt");
    
    else if(nFile==89)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File89-Pareto-J240-M80-R1-159-a1.2.txt");
    
    
    else if(nFile==90)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File90-worst-J181-M90-R90-179.txt");
    else if(nFile==91)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File91-semi-worst-J270-M90-R1-179.txt");
    else if(nFile==98)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File98-Pareto-J270-M90-R1-179-a1.3.txt");
    
    else if(nFile==99)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File99-Pareto-J270-M90-R1-179-a1.2.txt");
    
    else if(nFile==100)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File100-worst-J201-M100-R100-199.txt");
    else if(nFile==101)
        myfile.open("/wsu/home/ff/ff96/ff9687/WorstData/File101-semi-worst-J300-M100-R1-199.txt");
    else if(nFile==108)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File108-Pareto-J300-M100-R1-199-a1.3.txt");
    else if(nFile==109)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File109-Pareto-J300-M100-R1-199-a1.2.txt");
    
    
    
    else if(nFile==201)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File201-Pareto-J600-M200-R200-399-a1.2.txt");
    
    else if(nFile==202)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File202-Pareto-J600-M200-R200-399-a0.63.txt");
    else if(nFile==203)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File203-Pareto-J600-M200-R100-399-a0.63.txt");
    else if(nFile==204)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File204-Pareto-J600-M200-R1-399-a0.63.txt");

    else if(nFile==208)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File208-Pareto-J600-M200-R1-399-a1.3.txt");

    else if(nFile==209)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File209-Pareto-J600-M200-R1-399-a1.2.txt");
   else if(nFile==300)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File300-Pareto-J1000-M200-R100-399-a1.2.txt");    


  else if(nFile==308)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File308-Pareto-J900-M300-R1-599-a1.3.txt");

  else if(nFile==408)
        myfile.open("/wsu/home/ff/ff96/ff9687/ParetoData/File408-Pareto-J1200-M400-R1-799-a1.3.txt");



	if(!myfile)
	{	
		cout << "Error: file could not be opened"  << endl;
    //    outputFile << "Error: file could not be opened"  << endl;
		exit (EXIT_FAILURE);
	}

	myfile>>nJobs>>nMachines;
	for(int i=0; i<nJobs; i++)
	{
		int p;
		myfile>> p;
		ProcTimeJob.push_back(p);
	}
    
   /* outputFile <<"File "<< nFile << "is read."<<endl;
    outputFile<<"Number of Jobs : "<< nJobs << endl;
    outputFile<<"Number of Machines : "<< nMachines << endl;
    */
	cout<<"File "<< nFile << "is read."<<endl;
	cout<<"Number of Jobs : "<< nJobs << endl;
	cout<<"Number of Machines : "<< nMachines << endl;
	// printing Job processing which been read form the file
   /* outputFile <<"Job Processing times : "<< endl;*/
	//cout<<"Job Processing times : "<< endl;
	//printFun(ProcTimeJob);

}


