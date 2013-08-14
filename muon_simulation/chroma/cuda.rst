Cuda Threads and Blocks
=========================

* http://www.pgroup.com/lit/articles/insider/v2n1a5.htm

An Introduction to GPU Computing and CUDA Architecture 
Sarah Tariq, NVIDIA Corporation 

* http://on-demand.gputechconf.com/gtc-express/2011/presentations/GTC_Express_Sarah_Tariq_June2011.pdf

N-parallel invokations::

   add<<< N, 1 >>>(); 


#. each parallel invocation of add() is referred to as a block 
#. the set of blocks is referred to as a grid 
#. each invocation can refer to its block index using blockIdx.x 

::

    __global__ void add(int *a, int *b, int *c) { 
      c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]; 
     } 



Host/device coordination
---------------------------

Kernel launches are asynchronous, control returns to the CPU immediately 



Blocks and/or Threads
-----------------------

* a block can be split into parallel threads


Using parallel blocks::

    add<<<N,1>>>(d_a, d_b, d_c); 

    __global__ void add(int *a, int *b, int *c) { 
     c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]; 
    } 


Using parallel threads::

    add<<<1,N>>>(d_a, d_b, d_c); 

    __global__ void add(int *a, int *b, int *c) { 
        c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x]; 
    } 


blocks and threads::


    __global__ void add(int *a, int *b, int *c) { 
     int index = threadIdx.x + blockIdx.x * blockDim.x; 
     c[index] = a[index] + b[index]; 
    } 
    

    // handle arbitrary problem size
    __global__ void add(int *a, int *b, int *c, int n) { 
     int index = threadIdx.x + blockIdx.x * blockDim.x; 
     if (index < n) 
         c[index] = a[index] + b[index]; 
    } 
     
     

main for blocks and threads::

    #define N (2048*2048) 
    #define THREADS_PER_BLOCK 512 
    int main(void) { 
      int *a, *b, *c;   // host copies of a, b, c 
      int *d_a, *d_b, *d_c;  // device copies of a, b, c 
      int size = N * sizeof(int); 
       
      // Alloc space for device copies of a, b, c 
      cudaMalloc((void **)&d_a, size); 
      cudaMalloc((void **)&d_b, size); 
      cudaMalloc((void **)&d_c, size); 
     
      // Alloc space for host copies of a, b, c and setup input values 
      a = (int *)malloc(size); random_ints(a, N); 
      b = (int *)malloc(size); random_ints(b, N); 
      c = (int *)malloc(size);      

    // Copy inputs to device 
      cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); 
      cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); 
     
      // Launch add() kernel on GPU 
      //add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c); 
      add<<<(N + M - 1)/M,M>>>(d_a, d_b, d_c, N); 


      // Copy result back to host 
      cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost); 
     
      // Cleanup 
      free(a); free(b); free(c); 
      cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); 
      return 0; 
     } 



Why Bother with Threads? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike parallel blocks, threads have mechanisms to: Communicate, Synchronize 

* within a block, threads share data via shared memory
* Extremely fast on-chip memory, user-managed 
* Declare using `__shared__`, allocated per block 
* Data is not visible to threads in other blocks 

* use `__syncthreads();` as barrier when using `__shared__`



Launch N blocks with M threads per block with::

    kernel<<<N,M>>>(...)

* Use blockIdx.x to access block index within grid 
* Use threadIdx.x to access thread index within block 

* grid/block/thread

