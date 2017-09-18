// nvcc add.cu -run && rm a.out

#include <stdio.h>
#include <assert.h>


__global__ void add(int *a, int *b, int *c) 
{ 
    *c = *a + *b;
}



int main(void) 
{
  int a, b, c; // host 
  int *d_a, *d_b, *d_c; // pointers to device memory 

  int size = sizeof(int);
 
   // Allocate space for device copies of a, b, c
   cudaMalloc((void **)&d_a, size); 
   cudaMalloc((void **)&d_b, size); 
   cudaMalloc((void **)&d_c, size);
      

   // Setup input values
   a = 2; 
   b = 7;

   int c_expect = a + b ; 
 

   // Copy inputs to device
   cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice); 
   cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

   // Launch add() kernel on GPU
   add<<<1,1>>>(d_a, d_b, d_c);

   // Copy result back to host
   cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);


   printf(" a %d b %d c %d  \n", a, b, c );

   assert( c == c_expect );  
   

   // Cleanup
   cudaFree(d_a); 
   cudaFree(d_b); 
   cudaFree(d_c); 
   return 0;

}


