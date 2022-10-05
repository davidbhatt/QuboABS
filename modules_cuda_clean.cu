#include <stdio.h>

__device__ void fchangef(const float *qubo,bool *sol,int fbit,int bit,int qubo_size,float*dxo,bool solfbit);

__device__ int hamming(const bool *sol, const bool *solp, int nNodes);
__device__ int selectg(bool *sol,bool *solp,float *dxo,int nNodes,bool &solfbit);
__device__ float GetQubo(const float *qubo, int nNodes,int row, int col);
__device__ void straightSearch(float *qubo,bool *sol,float *dxo,bool *solp,bool *solb,float &energy,float &best,int nNodes);
__device__ int d_select(bool *sol,const int off,int l,float *dxo,int nNodes,bool &solfbit);
 __device__ void simple_localsearch(bool *sol,float *dxo,float *qubo,int nNodes,float &energy,float &best,bool *solb);

//variables

__device__ int count=0;
__device__ static const int gridSize =4 ;  //gridDIm
__device__ static const int blockSize =4 ; //blockDim
__device__ static const int nodes =16 ;   //nNodes
__shared__ float energy;
__shared__ float best;
__shared__ bool solb[nodes];
///////////////////
/////////////////////test functions/////////////////////////
__shared__ bool solt[nodes]; 


__global__ void search(float *qubo,int nNodes,int nsol,bool *d_tbuffer,bool *d_sbuffer,float *d_epool,int iterations,int *d_count,int *d_tcount)
{
	__shared__ int tcount_old[gridSize];



  int ibx=blockIdx.x; //block id
  int nblocks=gridDim.x;  //no of blocks 
  int idx = threadIdx.x; //thread id
  const int nb=nNodes/blockDim.x; //no of bits per thread
  int is=idx*blockDim.x;  //start index

  bool solc[blockSize];	
  float dxo[blockSize];	
  energy=0.0;
  //initiliazation 
  for (int i = is; i < is+nb;i++) 
    {
      solc[i-is]=0;
      dxo[i-is]=GetQubo(qubo,nNodes,i,i);  //for current solution
    }
  __syncthreads();
  if(idx==0) tcount_old[blockIdx.x]=0;
  while(count<iterations)
    {

	{
	count++;

	  //get the target solution

	  for(int i=is;i<is+nb;i++)
	    {
	      solt[i]=d_tbuffer[ibx*nNodes+i];
	    }
	  __syncthreads();

	  //reset best solution
	  for (int i = is; i < is+nb;i++) 
	    {
	      solb[i]=0;
	    }
	  best=0.0;
	  __syncthreads();

	  //perform straight search from current solution to target solution
	 straightSearch(qubo,solc,dxo,solt,solb,energy,best,nNodes);
	  
	  __syncthreads();
	  
	  if(idx==0)
	  {
		  printf("energy after completion oif straight search is %f on %d\n",energy,blockIdx.x);
		  printf("best energy after completion of straight search is %f on %d \n",best,blockIdx.x);
	  }
	  	  __syncthreads();

	  //perform localsearch from solc to another solc' here solc
	simple_localsearch(solc,dxo,qubo,nNodes,energy,best,solb);
	  
	 		__syncthreads();
	  //insert the best solution into the pool
	  for(int i=is;i<is+nb;i++)
	    d_sbuffer[ibx*nNodes+i]=solb[i];
	  __syncthreads();
	  if(idx==0)
	    d_epool[ibx]=best;
	  //inform the cpu about generation of new solution by increasing the counter
	  if(idx==0)
	  {
	    d_count[ibx]++;
		printf("new sol generated on device and stored in buffer for block %d \n",ibx);
	  }
	  if(idx==0)
	    {
	      printf(" best energy from device on %d is %f \n",ibx,best);
	      //printf("solb from device\n");


	    }
	  __syncthreads();
	  if(idx==0)	tcount_old[blockIdx.x]=d_tcount[blockIdx.x];

	}
    }
  delete[] solc;
  delete[] dxo;
  __syncthreads();
  
}

//fbit flipped bit
//dxo previous change in objective function
//bit is to be flipped bit
__device__ void fchangef(const float *qubo,bool *sol,int fbit,int bit,int qubo_size,float*dxo,bool solfbit)
{
	int idx = threadIdx.x; // thread id
	int is=idx*blockDim.x;  //start index

  //calculate dxo
  int k=fbit;
  int i=bit;
  if(k==i)
    {
      dxo[i-is]=-dxo[i-is];
    }
  else
    {
      if(solfbit==sol[i-is])
	{dxo[i-is]=dxo[i-is]+(GetQubo(qubo,qubo_size,i,k)+GetQubo(qubo,qubo_size,k,i));}
      else
	{dxo[i-is]=dxo[i-is]-(GetQubo(qubo,qubo_size,i,k)+GetQubo(qubo,qubo_size,k,i));}
    }
}

/***********************************************/

//straight search algorithm5
__device__ void straightSearch(float *qubo,bool *sol,float *dxo,bool *solp,bool *solb,float &energy,float &best,int nNodes)
//global 
{

  int idx = threadIdx.x; // thread id
  int is=idx*blockDim.x;  //start index
  int nb=nNodes/blockDim.x;  //bits per thread
  int a,b;
  int hd=hamming(sol,solp,nNodes);
  __syncthreads();
  //solution of flipped bit shared;
  __shared__ bool solfbit;
  //loop hamming distances reduces every step
  best=energy;
  //int i=0;
bool ans=true;
  for(int i=hd;i>0;i--)
    {
	 __syncthreads();

      a=selectg(sol,solp,dxo,nNodes,solfbit); //greedy search
		  __syncthreads(); 
	      for(b=is;b<is+nb;b++)
	{
	  if(b!=a)
	    {
	      fchangef(qubo,sol,a,b,nNodes,dxo,solfbit);
	    }
	}
      if(idx==(int)(a/nb))
	{
	  energy=energy+dxo[a-is];
	  dxo[a-is]=-dxo[a-is];
	  sol[a-is]=1-sol[a-is];
	}
    __syncthreads();
      if((energy<best))
	{
	  //copy the best solution
	  for(int c=is;c<is+nb;c++)
	    {
	      solb[c]=sol[c-is];
	    }
	  __syncthreads();
	  
	  if(idx==0)
	    best=energy;
	}
    }
  
}

/***************************************/
 __device__ void simple_localsearch(bool *sol,float *dxo,float *qubo,int nNodes,float &energy,float &best,bool *solb)
  {
	int ibx=blockDim.x;
	int idx = threadIdx.x;
    int nb=nNodes/blockDim.x;
    int is=idx*blockDim.x;  //start index
		int a,b;
	int offset=4; 
	int length=4; //sa temperature
	int iterations=2*nNodes;
	int im;  //index of max energy
	float en=best;  //energy local best
	 __shared__ bool solfbit;
	 __shared__ float dxoa;

	for(int kk=0;kk<iterations;kk++)
		  {
			  //defualt condition if local minima not able to find
			  im=-1;
			a=d_select(sol,offset,length,dxo,nNodes,solfbit);
				__syncthreads();

			//*if(idx==0) printf(" selection %d with offset %d \n",a,offset);
			offset=(offset+length)%nNodes;
		en=best; //set comparison value for energy 
				for(b=is;b<is+nb;b++)
		{
			if(b==a) {dxoa=dxo[a-is];}

			if(b!=a)
		{
	fchangef(qubo,sol,a,b,nNodes,dxo,solfbit);
		}
}
		__syncthreads();  //to update value of dxoa at all threads
		for(b=is;b<is+nb;b++)
		{

			if(b!=a)
		{

	//track change in energy and find minimum
		  if(energy+dxoa+dxo[b-is]<en)
	    {
	      im=b;
	      en=energy+dxo[b-is]+dxoa;   
	    }
	}
	}
__shared__ float min[blockSize];
  __shared__ int imin[blockSize];
  min[idx]=en;
  imin[idx]=im;
	__syncthreads();
	 //global min
  for (int size = blockDim.x/2; size>0; size/=2) 
    { //uniform
      if (idx<size)
	{
	  if(min[idx]>min[idx+size])
	    {
	      min[idx]=min[idx+size];
	      imin[idx]=imin[idx+size];
	    }
	}
      __syncthreads();
    }
		//update the best solution if new sol found
if(imin[0]!=-1)
{
	if(idx==(int)(imin[0]/nb))
	{
	best=min[0];
	  __syncthreads();
	}

				for(int c=is;c<is+nb;c++)
				  {
					  if(c==a | c==imin[0])  	//flip x at imin[0] and at a
					  {
						  solb[c]=1-sol[c-is];
					  }
					  else 
					  {solb[c]=sol[c-is];}
				  }
	__syncthreads();
}	
	 if(idx==(int)(a/nb))
	 {
	energy=energy+dxo[a-is];
	dxo[a-is]=-dxo[a-is];
	sol[a-is]=1-sol[a-is];
	 }
	 
__syncthreads();
	}

}
/***********************************************/

  		__device__ int d_select(bool *sol,const int off,int l,float *dxo,int nNodes,bool &solfbit)
			  {
				   int idx = threadIdx.x; // thread id
					int is=idx*blockDim.x;  //start index
					int nb=nNodes/blockDim.x;  //bits per thread
			    //find index with minimum value of energy change
				  float en;
				    int im=-1;  //local min
 __shared__ float min[blockSize];
  __shared__ int imin[blockSize];
//find locla min				 
  en=10000000000.00;
  //printf("selec tion dxos\n");
  for(int i=off;i<off+l;i++)
  {
	  if(i>=is && i<is+nb)
	  {
		//  printf("dxo val %f at %d on %d \n",dxo[i-is],i,idx);
		if(dxo[i-is]<en)
		{
			en=dxo[i-is];
			im=i;
		}
	  }	    
		  
} 
  min[idx]=en;
  imin[idx]=im;
  __syncthreads();

  //global min
  for (int size = blockDim.x/2; size>0; size/=2) 
    { //uniform
      if (idx<size)
	{
	  if(min[idx]>min[idx+size])
	    {
	      min[idx]=min[idx+size];
	      imin[idx]=imin[idx+size];
	    }
	}
	__syncthreads();
    
    }
	__syncthreads();
		//update solution at selected value
	if(threadIdx.x==(int)(imin[0]/nb))
	{
		solfbit=sol[imin[0]-is];
	}
	__syncthreads();
		  
	return imin[0];		  
			  
}

//hamming distance

__device__ int hamming(const bool *sol, const bool *solp, int nNodes)
  {
    int idx = threadIdx.x;
    int nb=nNodes/blockDim.x;
    int is=idx*blockDim.x;  //start index
    
    int ans=0;
    
    for(int i=is;i<is+nb;i++)
      {
	if(sol[i-is]!=solp[i])
	  ans++;
      }
    ///add all hamming distancees across threads
   __shared__ int r[blockSize];
    r[idx] = ans;
    __syncthreads();
    
    for (int size = blockDim.x/2; size>0; size/=2) { //uniform
      if (idx<size)
	r[idx] += r[idx+size];
      
      __syncthreads();
    }
    ans=r[0];
    return ans;  
  }

//cuda selection algorithm
__device__ int selectg(bool *sol,bool *solp,float *dxo,int nNodes,bool &solfbit)
{
  int idx = threadIdx.x; // thread id
  int is=idx*blockDim.x;  //start index
  int nb=nNodes/blockDim.x;  //bits per thread
  int im=0;  //local min
  //find index with minimum value of energy change
  float en;
  //find locla min				 
  en=10000000000.00;
  
  for(int i=is;i<is+nb;i++)
    {
      if(sol[i-is]!=solp[i])
	{
	  if(dxo[i-is]<en)
	    {
	      im=i;
	      en=dxo[i-is];   
	    }
	}
    }
  __shared__ float min[blockSize];
  __shared__ int imin[blockSize];
  min[idx]=en;
  imin[idx]=im;
  __syncthreads();
  //gloabl min
  for (int size = blockDim.x/2; size>0; size/=2) 
    { //uniform
      if (idx<size)
	{
	  if(min[idx]>min[idx+size])
	    {
	      min[idx]=min[idx+size];
	      imin[idx]=imin[idx+size];
	    }
	}
      __syncthreads();
    }
	//update solution at selected value
	if(idx==(int)(imin[0]/nb))
		solfbit=sol[imin[0]-is];
	__syncthreads();
  //if (idx == 0)
  return imin[0];					  
}

/***********/		
__device__ float GetQubo(const float *qubo, int nNodes,int row, int col)
{
  return qubo[row * nNodes + col];
};
/***********/	