//straight search algorithm5
 #include <stdio.h>
#include <iostream>
#include <string.h>
#include <vector> 
#include <algorithm>
#include <utility>
#include <random>
#include <iterator>
#include <iostream>
#include "host.cuh"
#include <iostream>
using namespace std;


float evaluate(bool *sol,float **qubo,int qubo_size) //use for first or last time
{
  float energy1=0.0;
  for(int i=0;i<qubo_size;++i)
	{
	  for(int j=0;j<qubo_size;++j)
	    {
	      energy1+=qubo[i][j]*sol[i]*sol[j];
	    }
	}
  return energy1;
 // cout<<energy1<<"in evaluate" <<endl;
}

//extra host functions

//Neighboorhood based 1-flip moves
//fast calculations of objective function
__host__ void fchange(float **qubo,bool *sol,int bit,int qubo_size,float*dxo)
{
  
  //calculate dxo
  dxo[bit]=qubo[bit][bit];
  for (int j = 0; j < qubo_size; ++j)
    {
      if(j!=bit &&sol[j]==1) //ignore zero values instead of multiplication
	{
	  dxo[bit]+=qubo[bit][j]+qubo[j][bit]; //row+column
	}
		
    }
  if(sol[bit]==1) dxo[bit]=-dxo[bit];  //sign change delta
  //flipping
  //sol[bit]=1-sol[bit];
  //update the flipfrequecy
  //flipfreq[bit]+=1;
  
}
__host__ float h_straightSearch(float **val,bool *sol,float *dxo,bool *solp,bool *solb,float &energy,int nNodes)
{
	float best;
	best=energy;
int a,b;
int ops;
int hd=h_hamming(sol,solp,nNodes);
//while(!equalQ(sol,solp,nNodes))
	for(int i=hd;i>0;i--)
	{
		cout<<"hamming distance "<<h_hamming(sol,solp,nNodes)<<" "<<i<<endl;
			a=h_selectg(sol,solp,dxo,nNodes); //greedy search
		  
		for(b=0;b<nNodes;b++)
		{
			if(b!=a)
	{
	h_fchangef(val,sol,a,b,nNodes,dxo,dxo);
	}
		}
	energy=energy+dxo[a];
	dxo[a]=-dxo[a];
	sol[a]=1-sol[a];
	ops++;
		
		if((energy<best))
			  {
				//copy the best solution
				  for(int c=0;c<nNodes;c++)
				  {
						  solb[c]=sol[c];
				  }
				   best=energy;
				  cout<<"best "<<best<<endl;
			  }
	}
	return best;
	//	cout<<"operations"<<ops<<endl;

	
}


//selection algorithm
			 __host__ int h_selectg(bool *sol,bool *solp,float *dxo,int nNodes)
			  {
				  int ans=0;
				  //find index with minimum value of energy change
				  float en=10000000000.00;
				  for(int i=0;i<nNodes;i++)
				  {
					  if(sol[i]!=solp[i])
					  {
					  if(dxo[i]<en)
					  {
						  ans=i;
						  en=dxo[i];
					  }
					  }
				  }	  
				return ans;					  
			  }
			  



 __host__ int h_hamming(bool *sol, bool *solp, int nNodes)
  {
	  int ans=0;
	  for(int i=0;i<nNodes;i++)
	  {
		  if(sol[i]!=solp[i])
			  ans++;
	  }
	  return ans;
	  
  }
  
  //fbit flipped bit
//dxo previous change in objective function
//bit is to be flipped bit
__host__ void h_fchangef(float **qubo,bool *sol,int fbit,int bit,int qubo_size,float*dxo1,float*dxo)
{
  
  //calculate dxo
  int k=fbit;
  int i=bit;
  if(k==i)
{
dxo[i]=-dxo1[i];
}
else
{
if(sol[k]==sol[i])
{dxo[i]=dxo1[i]+(qubo[i][k]+qubo[k][i]);}
else
{dxo[i]=dxo1[i]-(qubo[i][k]+qubo[k][i]);}
}
}
  bool equalQ(bool *sol,bool *sol1,int size)
  {
	  bool ans=true;
	  for(int i=0;i<size;i++)
		  ans&=(sol[i]==sol1[i]);
	  return ans;
  }
  
  ///genetic algorithm

//takes two solution from the pool and generates newsolution by performing crossover and mutation. the no of mutions is specified by numts
__host__ void GAreprod(bool*solpool,int nsol,bool *newsol,int nNodes,int nmuts)

{
	std::random_device rd;
    std::mt19937 g(rd());
	vector<int>slist;
	//copy solution 1
	int i,id,istart;
	for(i=0;i<nsol;i++)
		slist.push_back(i);
	std::shuffle(slist.begin(), slist.end(), g);
//take 2 form this list
	for(i=0;i<nNodes;i++)
		istart=slist[0]*nNodes;
		newsol[i]=solpool[istart+i];
	//crossover
	for(i=0;i<nNodes;i++)
	{
	id=rand()%2;
	if(id==1)
		istart=slist[1]*nNodes;
		newsol[i]=solpool[istart+i];
	}
	//create mutation
	vector<int>ind;	
	for(i=0;i<nNodes;i++)
		ind.push_back(i);
    std::shuffle(ind.begin(), ind.end(), g);
	//take first nmuts indices shuffled randomly
	for(i=0;i<nmuts;i++)
		id=ind[i];
		newsol[id]=1-newsol[id];
}


///subroutines for bianry search and replACE
///it gives the insertion point of a sorted array of floats
__host__ int binarySearch(float arr[], int size,float item) {
int il,ih,im;
il=0;ih=size-1;
if(item>arr[ih]) ///greater than max value of pool
{ih=-1;}
else
{
if(item<arr[il])
{ih=0;}
else if(item>arr[ih])
{ih=size;}
else
{
while(ih-il>1)
{
im=(il+ih)/2;
if(item<arr[im])
	ih=im;
else
	il=im;
}
}
}
return ih;
}

//replace ith entry of arr with value and removes the last entry
//changes the list of solution indexes
__host__ void replaceg(float arr[],int size,int id,float value,int *idsol)
{
	float max=arr[size-1];
	int maxid=idsol[size-1];
if(value<max)  //reject greater than energy pool
{	
  if(id<=size-1)
    {
      for(int i=size-1;i>id;i--)
	{
	  arr[i]=arr[i-1];
	  idsol[i]=idsol[i-1];
	  //shift the index values also
	  
	}
    }
  arr[id]=value;
  idsol[id]=maxid; //replace last value with the 
}
}
__host__ void initSol(bool *h_solpool,int nNodes,float *e_pool,int nsol,float **qubo)
{
	bool **sol_pool;
	sol_pool=new bool * [nsol];
		for(int i=0;i<nsol;i++)
	{
		sol_pool[i]=new bool [nNodes];
	}
	//initialize the solution pool with random values
	//srand(time(NULL)); 
	srand(10);
	for (int i=0;i<nsol;i++)
	{
		for(int j=0;j<nNodes;j++)
			sol_pool[i][j]=rand()%2;
		e_pool[i]=evaluate(sol_pool[i],qubo,nNodes);
	}
	//merge solutions in a single pointer
	int id;
		for (int i=0;i<nsol;i++)
		{
			for(int j=0;j<nNodes;j++)
			{
				id=i*nNodes+j;
				h_solpool[id]=sol_pool[i][j];
			}
		}

}
///index siorting
template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
};
//sorts energypool and generates the index for solution 
__host__ void sortSolution(float *e_pool,int nsol,int *ids)
{
	//int *ids=new int [nsol];  ///index of solution in increasing order of energies
vector <float>v;
for(int i=0;i<nsol;i++)
	v.push_back(e_pool[i]);
//get indexes
int j=0;
for (auto i: sort_indexes(v)) {
	ids[j]=i;
  j++;
}
	//sort energies
	for(int j=0;j<nsol;j++)
	{e_pool[j]=v[ids[j]];
	}
	
}

//inserts solution energy in the e_pool and updates the index for the solution 
//replaces the worst solution with the new best solution in hte solution pool
//coreespondingly updates the energy also  
__host__ void insertSol(float entry,float *e_pool,int nsol,int *ids,bool *bestsol,bool *solpool,int nNodes)
{ 
    int id=binarySearch(e_pool, nsol,entry);
	//skips if the energy is greater than that maximum value in the pool
	if(id!=-1)
	{
		replaceg(e_pool,nsol,id,entry,ids);
		
  //insert the new solution in the solutionbuffer at location ids[id]  worst energy
	insert(bestsol,solpool,ids[id],nNodes);
	}
}

//insert new solution in the buffer
__host__ void insert(bool *newsol,bool *solpool,int point,int nNodes)
{
	int istart=point*nNodes;
	for(int i=0;i<nNodes;i++)
	{
		solpool[i+istart]=newsol[i];
		
	}
	
	
}
