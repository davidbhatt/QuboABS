#include <stdio.h>
#include <iostream>
#include <tuple>
#include <vector> 
#include <algorithm>
using namespace std;
__host__ void read_qubo(FILE *inFile,int &nmin,int &nmax,int &nNodes)
{
  char a[1000];
  if (inFile == NULL)
    {
      fprintf(stderr,
	      "\n\t%s error -- no input file (-i option) specified"
	      "\n\n",
	      "qbsolv");
      exit(9);
    }
  fscanf(inFile,"%s",a); //ignore first line
  
  int f1,f2;
  float f3;
  vector <int> entries;
  while (fscanf(inFile, "%d%d%f\n", &f1, &f2,&f3) == 3)
    {
      //      printf("%d %d %f\n", f1,f2,f3);
      entries.push_back(f1);
    }
  nmax=*max_element(entries.begin(), entries.end());
  nmin=*min_element(entries.begin(), entries.end());
 nNodes=nmax-nmin+1;
cout<<"found nodes "<<nNodes<<"("<<nmin<<","<<nmax<<")"<<endl;
}
__host__ void fill_qubo(FILE *inFile,float **val,int nmin)
{
  char a[100];
  int f1,f2;
  float f3;
  rewind(inFile);
  fscanf(inFile,"%s",a); //ignore first line
  while (fscanf(inFile, "%d%d%f\n", &f1, &f2,&f3) == 3)
    {
      //  printf("%d %d %f\n", f1,f2,f3);
      //  *(*(val + f1 - 1) + f2 - 1)=f3
      val[f1-nmin][f2-nmin]=f3;
      //	  val[f2-nmin][f1-nmin]=f3/2;
  }
  
}
//function to make matrix lower triangular to make use of fast function calcuations for 1-flip move
__host__ void LowerTriangulize(float **qubo,int qubo_size)
{
		//make qubo matrix lower triangiular
	for (int i = 0; i <qubo_size; ++i)
	{
		for (int j = 0; j < qubo_size; ++j)
		{
			if(i<j)
			{
				qubo[i][j]=qubo[i][j]+qubo[j][i];
			}else if(i>j)
			{
			qubo[i][j]=0.0;
			}
		}
	}
}
