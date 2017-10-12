
// nvcc hello.cu -run && rm a.out  

#include <stdio.h>

__global__ void mykernel(void) 
{
    printf("Hello World (from mykernel)!\n");
}


int main(void) 
{
    mykernel<<<1,1>>>();
    printf("Hello World!\n");


    cudaDeviceSynchronize();  

    // Without the sync the process will typically terminate before 
    // any output stream gets pumped out to the terminal, so will not see
    // Hello World (from mykernel) 

    return 0;
}

