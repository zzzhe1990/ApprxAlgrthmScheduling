#ifndef PARALLEL_H
#define	PARALLEL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cmath> 
#include <fstream>
#include <sstream>
#include <cstring>
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
#define _HOST_DEBUG
//#define _DEVICE_DEBUG

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
	block(int m_size):mySUM(m_size){};
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
    //DynamicTable():elm(0, 0), optVector(0, 0), myOPT(0), mySum(0), myOptimalindex(0), myMinNSVector(0, 0), NSsubsets(0, vector<int>(0, 0)), Csubsets(0, vector<int>(0, 0)){};
    //~DynamicTable();
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
    //FinalTableINFO():Nvector(0, 0), OPTtable(0), NSTableElements(0), AllTableElemets(0), Ttable(0), optimalValuesVector(0, 0), roundVecTable(0, 0), ShortJobsTable(0, 0), LongJobsTable(0, 0), LongRoundJobsTable(0, 0), roundCriteriaTable(0){};
	//~FinalTableINFO();
};

#endif
