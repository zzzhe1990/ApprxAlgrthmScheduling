//
//  GeneratingData.cpp
//  
//
//  Created by Laleh Ghalami on 8/13/16.
//
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <string>



using namespace std;

int Uniform(int LB,int UB);
int BoundedPareto(float alpha, float LB, float UB);
int Pareto(float alpha, float LB)
;



int main(void)
{
    int n,m;
    float alpha=0.9;
    int LB = 1;
    int UB=100;
    
    int seed = time(NULL);
    srand(seed);
    
    string str0=  "/Users/lalehghalami/Dropbox/Scheduling/ParetoData/File";
    string str1;
    string str2;
    string str3 ="-";
    string str4;
    string str5=".txt";
    
    int Machine[10]={100,200,300,400,500};

    
    int nFile=320;

    for (int i=0; i<5; i++) {
        m=Machine[i];
        for (int j=0; j<2; j++) {
            if(j==0)
                n=10*m;
            else if(j==1)
                n=20*m;
            
                for (int f=1; f<21; f++)
                {
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
                    ofstream myFile(str1);
               
                
                myFile << n << "\n" << m << "\n";
                
                
                    for(int r=0; r<n; r++)
                    {
                        int myParetoNumber;
                        myParetoNumber=BoundedPareto(alpha,LB,UB);
                        myFile << myParetoNumber << "\n";
                    }
                
                    
                }
            nFile++;
            
        }
    }
    

    return 0;
}



int Uniform(int LB,int UB)
{

    random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int>  distr(LB, UB);
    
    return distr(generator);
}


int Pareto(float alpha, float LB)
{
    int U2;
    float U;
    float R1;
    int R2;
    
    U2= rand() % 100;
    U= (float)rand()/((float)(RAND_MAX)+1);
    R1= (float)LB /(float)(pow(U,(float)1/alpha));
    R2= ceil(R1);
    return R2;
}




int BoundedPareto(float alpha, float LB, float UB)
{
    
    float U;
    float paretoTemp1;
    float paretoTemp2;
    int paretoNum;
    
    U = (float)rand()/((float)(RAND_MAX)+1);
    
    paretoTemp1= ( pow(UB,alpha) + U*pow(LB,alpha) - U*pow(UB,alpha) ) / ((pow(UB,alpha))*(pow(LB,alpha)));
    paretoTemp2 = (float)(pow(paretoTemp1,(float)(-1)/alpha));
    paretoNum= ceil(paretoTemp2);
    return paretoNum;
}




