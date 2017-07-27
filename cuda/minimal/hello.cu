
// nvcc -arch=sm_30 hello.cu -run ; rm a.out
// https://stackoverflow.com/questions/8003166/usage-of-printf-in-cuda-4-0-compilation-error

#include <stdio.h>

__global__ void helloCUDA(float f)
{
  printf("Hello thread %d, float(f)=%f\n", threadIdx.x, f);
}

__global__ void helloCUDA(double f)
{
  printf("Hello thread %d, double(e)=%e (g)%g (E)%E (G)%G \n", threadIdx.x, f, f, f, f );
}



int main()
{
  helloCUDA<<<1, 5>>>(1.2345f);

  helloCUDA<<<1, 5>>>(1.2345);


  cudaDeviceReset();
  return 0;
}
