//#include "Parallel.h"
#include "DPCUDA.h"

//  Global defination 
vector <DynamicTable> NSTableElements;
vector <DynamicTable> AllTableElemets;
vector<int> tempOptVector;
vector < FinalTableINFO > AllProbData;

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
//int nthreads,nthreads0;

//string str0=  "/Users/lalehghalami/Desktop/parallelCode/File";
string str0 = "/home/gomc/Desktop/ApproxAlgorthim/Git/ApprxAlgrthmScheduling/File";
//string str0=  "/wsu/home/ff/ff96/ff9687/ParetoData/File";
//string str0=  "/wsu/home/ff/ff96/ff9687/UniformData/File";
//string str0=  "/Users/lalehghalami/Dropbox/Scheduling/UniformData/File";
string str1;
string str2;
string str3 ="-";
string str4;
string str5=".txt";


//ofstream solution("AllInstances.xls");

ofstream solution("/home/gomc/Desktop/ApproxAlgorthim/Git/ApprxAlgrthmScheduling/PTAS-Results-T4F308-5.xls");


//ofstream solution("/wsu/home/ff/ff96/ff9687/Results-Pareto-22Sep/PTAS-Results-T4F308-5.xls");
//ofstream scheduleFile("/wsu/home/ff/ff96/ff9687/Results-8Sep/PTAS-Schedule-T16F40.xls");
//ofstream output("/wsu/home/ff/ff96/ff9687/Results-9Aug/Par-Output-File300.xls");

int main(int argc, char* argv[])
//int main()
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
    }
    
    //nFile=100;
    //error =0.3;
    //th=0;

    //nthreads0= Pow(2,th);
     
    solution << " nFile " << " \t" << "f" << "\t" << " njobs " << "\t" << " nMachines " << "\t" <<" Error  "  << "\t"  << " nThreads " << "\t"<< "LB0"<<"\t"<<"UB0"<<"\t " << "Num Short"<<"\t"<<"Num Long"<<"\t"<<"OPT"<<"\t"<<"makespan" << "\t" <<" Total Time getTimeofday"<<   "\t" << "Host Name" << endl;

  char hostname[HOST_NAME_MAX];
    if (! gethostname(hostname, sizeof hostname) == 0)
      perror("gethostname");

    while (nFile<302)
    {
        for (int f=1; f<2; f++)
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

	while(LB<UB){
		clearFun();
		
		BkID= Bk(LB,UB);
		cout << "BKID: " << BkID << ", LB: " << LB << ", UB: " << UB << endl;
        if (BkID==-1) {
			cout << "BkID==-1" <<endl;
            break;
        }
        
        //cout << "Check the line 255" << endl;
		//print(iwhile);
		if(OPT<=nMachines)
			UB=T;
		if(OPT>nMachines)
			LB=T+1;
		iwhile++;
	}

    
    //cout << "********************************************************"<<endl;
    //cout << "OUT of Bk while loop  "<<endl;
    //cout << "UB    "<< UB<<endl;
    //cout << "LB    "<< LB<<endl;
    //cout << "T1    "<< T<<endl;
   	double TT;
    TT=(LB+UB)/2;
    T=floor(TT);
    //cout << "T2    "<< T<<endl;
    //cout << "OPT   "<< OPT<<endl;
    //cout << "nMachines   "<< nMachines<<endl;
    //cout << "AllProbData.size()    "<< AllProbData.size()<<endl;
    
    
    //for (int ck=AllProbData.size()-1; ck >= 0; ck--)
   // {
     //   cout << "ck :   " << ck << endl;
       // cout << "AllProbData[ck].Ttable    "<< AllProbData[ck].Ttable <<endl;

    //}


    
    cout << "optIndex    "<< optIndex<<endl;


    
    vector<int> temp;
	for(int i=0;i<nMachines;i++)
	{
		OptimalSchedule.push_back(temp);
	}
	for(int i=0;i<nMachines;i++)
	{
		machineTimes.push_back(0);
	}
    
    
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
    
    
/*    scheduleFile << "Fopt"<<"\t"<<Fopt<<endl;
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
    scheduleFile << endl;
  */  

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
		int index;
      //  cout << "RoundedOptimalSchedule" <<endl;
		for(int i=0;i<RoundedOptimalSchedule.size();i++)
		{
        //    cout << "RoundedOptimalSchedule[ " <<i<<" ]:    ";
			for(int j=0; j< RoundedOptimalSchedule[i].size();j++)
			{
          //      cout << RoundedOptimalSchedule[i][j]<<"  ";
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
            cout<<endl;
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

    
    //cout << "Machine Times" <<endl;
    //for (int ck=0; ck< machineTimes.size();ck++) {
    //    cout << machineTimes[ck] <<"  ";
   // }
    //cout<<endl;

	if(shortF.size()!=0){
		ListSchedulingFun(shortF,OptimalSchedule,machineTimes,OPT);
      //  cout << "Short jobs has been aded"<<endl;
	}

    //cout << "Machine Times" <<endl;
    //for (int ck=0; ck< machineTimes.size();ck++) {
      //  cout << machineTimes[ck] <<"  ";
    //}
    //cout<<endl;

	int makespan;
	makespan = max_element(machineTimes.begin(), machineTimes.end()) - machineTimes.begin();
	FinalMakespan = machineTimes[makespan];
    
    
    
    cout << "Final OptimalSchedule" << endl;
    //for (int q=0; q<OptimalSchedule.size(); q++)
   // {
     //   for (int e=0; e<OptimalSchedule[q].size(); e++)
       // {
         //   cout << OptimalSchedule[q][e] << " ";
       // }
        //cout<< endl;
    //}
    
/*    scheduleFile << "Long Jobs:"<<endl;
    for (int i=0; i < longF.size(); i++)
    {
        scheduleFile << longF[i] << "\t";
    }
    scheduleFile << endl;
  */  
    
    cout << "FinalMakespan" << FinalMakespan<< endl;
    printFinalSchedule( OptimalSchedule, machineTimes, longF,shortF,Fopt);
    
    
	NSTableElements.clear();
	AllTableElemets.clear();
	tempOptVector.clear();
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
        cout << " LongJobs.size() " <<LongJobs.size()<<endl;
        OPT=DPFunction2(Ntemp);
        return 0;
    }
    else
    {cout << " LongJobs.size() " <<LongJobs.size()<<endl;
        return -1;}
	
	
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
    
    InitGPUData(AllTableElemets.size(), Cwhole.size(), powK, LongJobs.size(), AllTableElemets, &zeroVec[0], &roundVec[0], &counterVec[0]);
    /*
	gpu_DP(AllTableElemets, dev_ATE_elm, dev_counterVec, dev_roundVec, T, k, powK, 
		   AllTableElemets.size(), dev_ATE_Csubsets, dev_ATE_NSsubsets, 
		   dev_ATE_NSsubsets_size, Cwhole.size(), dev_zeroVec, dev_ATE_optVector, 
		   dev_ATE_optVector_size, dev_ATE_myOPT, dev_ATE_myOptimalindex, 
		   dev_ATE_myMinNSVector, it, ss, NS, maxSumValue, counterVec);
	*/
    gpu_DP(AllTableElemets, T, k, powK, AllTableElemets.size(), 
		   Cwhole.size(), maxSumValue, counterVec, LongJobs.size());

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

    cout << "dpoptimal" <<dpoptimal << endl;
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

	sumVec.clear();
	Cwhole.clear();
  
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


