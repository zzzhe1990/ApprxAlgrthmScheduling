#ifndef PARALLEL_H
#define	PARALLEL_H

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
#include <time.h>
#include <omp.h>


#define SPLIT
//#define DEBUG

using namespace std;

class dim
{
public:
	int weit;
	int index;
	bool operator< (const dim &a) const{
		return (weit < a.weit);
	}
};


class block
{
public:
	vector<int> elm;
	int mySUM;
	bool operator < (const block &a) const{
		return (mySUM < a.mySUM);
	}
};

class DynamicTable
{
public:
	vector <int> elm;
	vector < vector <int> > NSsubsets;		//it is the same as NMinusStemp
	vector < vector <int> > Csubsets;		//for each configuration, they have their own Csubset.
	vector <int> optVector;			//optimal of the subproblems of 
	//vector <int> CsumOverVec;
	int myOPT;
    int mySum;
    int myOptimalindex;
    vector <int> myMinNSVector;
    bool operator< (const DynamicTable &a) const {
        return (mySum < a.mySum);
    }
};



class FinalTableINFO		//optimal schedule
{
public:
	vector<int> Nvector;
	int OPTtable;
	vector <DynamicTable> NSTableElements;
	vector <DynamicTable> AllTableElemets;			//vector V in the paper
	int Ttable;
	vector <int> optimalValuesVector;
    vector<int> roundVecTable;
    vector<int> ShortJobsTable;
    vector<int> LongJobsTable;
    vector<int> LongRoundJobsTable;
    double roundCriteriaTable;
};

#endif
