/************************************************************
This code solves the NP hard problem of QUBO (Quadratic unconstrained Binary Optimization) 
It follows the Adaptive Bulk Search Algorithm described in the paper 
Ryota Yasudo, Koji Nakano, Yasuaki Ito, Masaru Tatekawa, Ryota Katsuki, Takashi Yazane, and Yoko Inaba. 2020. 
Adaptive Bulk Search: Solving Quadratic Unconstrained Binary Optimization Problems on Multiple GPUs. 
In 49th International Conference on Parallel Processing - ICPP (ICPP '20). Association for Computing Machinery, New York, NY, USA, Article 62, 1–11.
 https://doi.org/10.1145/3404397.3404423
This is implemented in CUDA C++
to compile in linux use ./compile.sh after changing necessary compile flags
the options to be provided are
-i filename #compulsory options filename containing the qubo matrix in qubo format
#optional options include
-it  no of iterations ofr local search
-nsol no of parallel instances of solutions to generate

**********************************************************/
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <vector> 
#include <algorithm>
#include <utility>
#include <random>
#include <iterator>
#include <iostream>
#define the CUDA_API_PER_THREAD_DEFAULT_STREAM //enable concurrency
#include <cuda_runtime.h>
#include "host.cuh"

using namespace std;
#define BITS_PER_THREADS 4

///combined search function straight search+ local search
__global__ void search(float *qubo,int nNodes,int nsol,bool *d_tbuffer,bool *d_sbuffer,float *d_epool,int iterations,int *d_count,int *d_tcount);

int main(int argc, char *argv[])
{
  char *inFileName = NULL;
  FILE *inFile = NULL;
  //srand(time(NULL));
  srand(10);
  cout << "You have entered " << argc
       << " arguments:" << "\n";
  int niterations=1000;
  int nsol=4; //no of solutions in solution buffers
  int nmutations; //no of mutations
  for(int i=1;i<argc;++i)
    {
      cout << argv[i] << "\n";
      if(strstr(argv[i], "-i"))  //argv[i]=="-"&&argv[i][1]=="i")
	{
	  inFileName=argv[i+1];
	  break;
	}
      if(strstr(argv[i], "-it"))  //argv[i]=="-"&&argv[i][1]=="i")
	{
	  niterations=atoi(argv[i+1]);
	  break;
	}
      if(strstr(argv[i], "-nsol"))  //argv[i]=="-"&&argv[i][1]=="i")
	{
	  nsol=atoi(argv[i+1]);
	  break;
	}
    }
  
  printf("supplied file name is %s\n",inFileName);
  cout<<"iterations = "<<niterations<<endl;
  //read file
  inFile = fopen(inFileName, "r");
  
  int nmin,nmax,nNodes;
  read_qubo(inFile,nmin,nmax,nNodes);
  
  cout<<"found nodes "<<nNodes<<"("<<nmin<<","<<nmax<<")"<<endl;
  //fill qubo matrix
  float **val;
  val= new float *[nNodes];
  for(int i = 0; i <nNodes; i++) val[i] = new float[nNodes];
  fill_qubo(inFile,val,nmin);
  fclose(inFile);
  
  //parmateres
  
  nmutations=nNodes/4;
  //make qubo matrix lower triangular
  //	LowerTriangulize(val,nNodes);
  //initialization
  //solution pool
  bool *h_solpool=new bool [nsol*nNodes];
  float *h_epool=new float [nsol];
  //generate random initial solution buffer and calclaute their energies in h_epool
  initSol(h_solpool,nNodes,h_epool,nsol,val);

    //index vector
  int *h_indsol;
  h_indsol=new int [nsol];
  
  //sort solutions and get indices 
  sortSolution(h_epool,nsol,h_indsol);
  
  cout<<" energy after sorting from pool "<<endl;
  for(int i=0;i<nsol;i++)
    {
      cout<<h_epool[i]<<" "<<evaluate(h_solpool+h_indsol[i]*nNodes,val,nNodes)<<endl;
      for(int j=0;j<nNodes;j++)
	cout<<h_solpool[h_indsol[i]*nNodes+j];
      cout<<endl;
    }
  
  cudaError_t err;
  const size_t malloc_limit = size_t(1024) * size_t(1024) * size_t(1024);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, malloc_limit); 
  //allocate device memory
  ////device target buffer
  size_t size = nNodes *nsol* sizeof(bool); 
//target buffer
  bool *h_tbuffer=0;
  err=cudaMallocHost((void **)&h_tbuffer, size,cudaHostAllocWriteCombined);
  if(err!=cudaSuccess ) cout<<"unable to allocate host target buffer"<<cudaGetErrorString(err)<<endl;
  
  
  //solution energy buffer
  float *h_ebuffer;
  err=cudaMallocHost((void **)&h_ebuffer, nsol*sizeof(float));
  if(err!=cudaSuccess ) cout<<"unable to allocate host solution buffer"<<cudaGetErrorString(err)<<endl;
  
  
  //copy initial solution to target buffer
  for(int k=0;k<nsol*nNodes;k++)
    {
      h_tbuffer[k]=h_solpool[k];
    }
  
  //allocate 1 dimensional array for qubo to facilitate transfer
  float *qubo;
  qubo=new float [nNodes*nNodes];
  int id;
  for(int i=0;i<nNodes;i++)
    {
      for(int j=0;j<nNodes;j++)
	{
	  id=i*nNodes+j;
	  qubo[id]=val[i][j];
	}
    }
  
  
  bool *d_tbuffer=NULL;
  err=cudaMalloc((void **)&d_tbuffer, size);
  if(err!=cudaSuccess ) cout<<"unable to allocate target buffer"<<cudaGetErrorString(err)<<endl;
  
  ////device best solution buffer
  bool *d_sbuffer=NULL;
  err=cudaMalloc((void **)&d_sbuffer, size);
  if(err!=cudaSuccess ) cout<<"unable to allocate device solution buffer"<<cudaGetErrorString(err)<<endl;
  ////host best solution buffer to hold solutions and insert in the solution pool
  bool *h_sbuffer=0;
  err=cudaMallocHost((void **)&h_sbuffer, size);
  if(err!=cudaSuccess ) cout<<"unable to allocate host solution buffer"<<cudaGetErrorString(err)<<endl;
  
  // the device qubo matrix
  size_t size2d=nNodes*nNodes*sizeof(float);
  float *d_qubo = NULL;
  err=cudaMalloc((void **)&d_qubo, size2d);
  if(err!=cudaSuccess ) cout<<"unable to allocate qubo"<<cudaGetErrorString(err)<<endl;

  //copy from host to device
  err=cudaMemcpy(d_qubo, qubo, size2d, cudaMemcpyHostToDevice);
  if(err!=cudaSuccess ) cout<<"qubo copy failed "<<cudaGetErrorString(err)<<endl;
  
  
  err=cudaMemcpy(d_tbuffer, h_tbuffer, size, cudaMemcpyHostToDevice);
  if(err!=cudaSuccess ) cout<<"target buffer copy failed "<<cudaGetErrorString(err)<<endl;
  
	    // the device energy pool
  size_t sizeE=nsol*sizeof(float);
  float *d_epool ;
  err=cudaMalloc(&d_epool, sizeE);
  if(err!=cudaSuccess ) cout<<"unable to allocate energy pool"<<cudaGetErrorString(err)<<endl;
  /*****************************************************/
  //streams
  cudaStream_t stream1,stream2;
  cudaStreamCreate(&stream1); //d2h counter
  cudaStreamCreate(&stream2);  //h2d counter
  //stream for d2h
  cudaStream_t streamd2h,	streamh2d;
  cudaStreamCreate(&streamd2h);
  cudaStreamCreate(&streamh2d);
  // create cuda event handles
  cudaEvent_t start, stop,copy;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&copy);
  //run the actual function localsearch
  /**************/
  //counters
  //solution buffer count
  int *h_count = 0;
  err=cudaMallocHost((void **)&h_count, nsol*sizeof(int));
    if(err!=cudaSuccess ) cout<<"could not allocate h_count "<<endl;

  int *d_count = 0;
  err=cudaMalloc((void **)&d_count, nsol*sizeof(int));
      if(err!=cudaSuccess ) cout<<"could not allocate d_count "<<endl;

  //target buffer count
   int *h_tcount = 0;
  err=cudaMallocHost((void **)&h_tcount, nsol*sizeof(int),cudaHostAllocWriteCombined);
        if(err!=cudaSuccess ) cout<<"could not allocate h_tcount "<<endl;

  int *d_tcount = 0;
  err=cudaMalloc((void **)&d_tcount, nsol*sizeof(int));
          if(err!=cudaSuccess ) cout<<"could not allocate d_tcount "<<endl;

  //initialize
  int h_count_old[nsol];
  
  for(int i=0;i<nsol;i++)
    {
      h_count[i]=0;
      h_tcount[i]=1;
    }
  ////send to device
  cudaEventRecord(start, 0);
  
  cudaMemcpy(d_count,h_count,nsol*sizeof(int),cudaMemcpyHostToDevice);	
  cudaMemcpy(d_tcount,h_tcount,nsol*sizeof(int),cudaMemcpyHostToDevice);	 		
  
  /*************************************************************/
  cout<<"cuda local search test"<<endl;
  static const int nthreads=nNodes/BITS_PER_THREADS;
  ///combined search function straight search+ local search
  search<<<nsol,nthreads,0,0>>>(d_qubo,nNodes,nsol,d_tbuffer,d_sbuffer,d_epool,niterations,d_count,d_tcount);
  cout<<"test exited "<<endl;
  
  err = cudaGetLastError();
  
  if(err!=cudaSuccess ) cout<<"LOCAL SEARCH failed "<<cudaGetErrorString(err)<<endl;
  cudaEventRecord(stop, 0);
  /****event **************/
  //create event for copy purpose
  cudaEvent_t copys[nsol],copyt[nsol];
  for(int i=0;i<nsol;i++)
    {
      cudaEventCreate(&copys[i]);
      cudaEventCreate(&copyt[i]);
    }
  
  /****************************/
  //cpu code
  std::vector<int>rec;
  int counter=0;
  for(int k=0;k<nsol;k++) h_count_old[k]=h_count[k];
  while (cudaEventQuery(stop) == cudaErrorNotReady) 
    {
      //h_count++;
      
      cudaMemcpyAsync(h_count,d_count,nsol*sizeof(int),cudaMemcpyDeviceToHost,stream1);	
      cudaEventRecord(copy, stream1);
      
      if(cudaEventQuery(copy)==cudaSuccess)
	{
	  int newsols=0; //no of new solutions generated
	  for(int k=0;k<nsol;k++)
	    {newsols+=h_count[k]-h_count_old[k];}
	  rec.push_back(newsols);
	  if(newsols>0)
	    {
	      counter++;
	      cout<<counter<<" new sols found ="<<newsols<<endl;
	      cudaMemcpyAsync(h_sbuffer,d_sbuffer,nNodes*sizeof(bool),cudaMemcpyDeviceToHost,stream1);
		  cudaMemcpyAsync(h_ebuffer,d_epool,sizeof(float),cudaMemcpyDeviceToHost,stream1);
		  cudaEventRecord(copys[0], stream1);
	     
		int istart;
		//wait for completion of copy of solution from device
		do{} while(cudaEventQuery(copys[0])!=cudaSuccess);
	      //update solution pool
	      for(int k=0;k<nsol;k++)
		{
		  if((h_count[k]>h_count_old[k])&& (cudaEventQuery(copys[0])==cudaSuccess))
		    {
			  //update solution pool
			  istart=k*nNodes; 
			  insertSol(h_ebuffer[k],h_epool,nsol,h_indsol,h_sbuffer+istart,h_solpool,nNodes);
		      cout<<"new solutions inserted at "<<k<<endl;
		    }
		}
	      //generate new solutions
	      
	      for(int k=0;k<nsol;k++)
			{
			if(h_count[k]>h_count_old[k])
				{
				GAreprod(h_solpool,nsol,h_tbuffer+k*nNodes,nNodes,nmutations);
				cout<<"new sol generated for "<<k<<" "<<endl;
				}
			}
	      cout<<endl;
	      ///copy the new solutions to the device
	      for(int k=0;k<nsol;k++)
		{
		  if(h_count[k]>h_count_old[k])
		    {
		      istart=k*nNodes; 
		      cudaMemcpyAsync(d_tbuffer+istart,h_tbuffer+istart,nNodes*sizeof(bool),cudaMemcpyHostToDevice,stream2);
		      	      
		      h_tcount[k]++;
			cout<<"new sols sent for k ="<<k<<endl; 
			  //update the counter
			h_count_old[k]=h_count[k];

		      err = cudaGetLastError();
		      
		      if(err!=cudaSuccess ) cout<<"error  "<<cudaGetErrorString(err)<<endl;
		      
		      

		    }
		}
		for(int k=0;k<nsol;k++)
			cout<<" k= "<<k<<" count="<<h_tcount[k]<<endl;
				      //send counter also
		  cudaEvent_t ccount;
		  cudaEventCreate(&ccount);
			  cudaMemcpyAsync(d_tcount,h_tcount,nsol*sizeof(int),cudaMemcpyHostToDevice,stream2);
			  cudaEventRecord(ccount, stream2);
	    
	      err = cudaGetLastError();
	      
	      if(err!=cudaSuccess ) cout<<"copy failed "<<cudaGetErrorString(err)<<endl;
	      	do{}
			while(cudaEventQuery(ccount)!=cudaSuccess);
	      cout<<"copied data and counts to device successfully "<<endl;
	    }
	  //send the cpounter to device
	  
	}
    }
  
  //complete updating target buffer
  cudaStreamSynchronize(streamh2d);
  cudaStreamSynchronize(streamd2h);
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
  //finalize
  cudaDeviceSynchronize();
  
  //print the best solution
  cout<<"best solution found so far "<<h_epool[0]<<endl;
  int k=h_indsol[0];
  for(int i=0;i<nNodes;i++)
    {
      cout<<h_solpool[k*nNodes+i];
    }
  cout<<endl;
  cout<<" energy check: " <<endl;
  for(int i=0;i<nsol;i++)
    {
      k=h_indsol[i];
      cout<<h_epool[i]<<"  "<<k<<" "<<evaluate(h_solpool+k*nNodes,val,nNodes)<<endl;
    }
  
  printf("CPU generated new solution %d times and updated to target buffer\n", counter);
  cudaDeviceSynchronize();
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
   
  /********************************************/
  
  //free streams
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(streamd2h);
  cudaStreamDestroy(streamh2d);
  // Free device memory
  cudaFree(d_qubo);
  cudaFree(d_tbuffer);
  cudaFree(d_sbuffer);
  cudaFree(d_epool);
  cudaFree(d_count);
  cudaFree(d_tcount);
  //free host memory 
  cudaFreeHost(h_ebuffer);
  cudaFreeHost(h_sbuffer);
  cudaFreeHost(h_tbuffer);
  cudaFreeHost(h_count);
  cudaFreeHost(h_tcount);
  delete [] qubo;
  for(int i=0;i<nNodes;i++)
    delete [] val[i];
  delete [] val;
  delete h_solpool;
  delete h_epool;
  delete h_indsol;
  cudaDeviceReset();
  
  return 0;
}
