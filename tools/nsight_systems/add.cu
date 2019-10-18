// https://devblogs.nvidia.com/transitioning-nsight-systems-nvidia-visual-profiler-nvprof/
/**

   nsight-
   nsight-systems-nvtx-include-dir

   nsight- ; nvcc -o /tmp/add add.cu -L$(nsight-systems-nvtx-include-dir)


   nsys profile -o /tmp/add.profile --stats=true /tmp/add
   nsys profile -o /tmp/add.profile --stats=true --force-overwrite true /tmp/add 

    [blyth@localhost nsight_systems]$ l  /tmp/add.profile*
    -rw-r--r--. 1 blyth blyth 1372160 Sep 23 23:10 /tmp/add.profile.sqlite
    -rw-rw-r--. 1 blyth blyth  171480 Sep 23 23:10 /tmp/add.profile.qdrep



**/


#define USE_NVTX 1

// https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/

#ifdef USE_NVTX
#include "nvtx3/nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif



#include <iostream>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
 
int main(void)
{
  int N = 1<<20;
  float *x, *y;


   
  PUSH_RANGE("Allocate Unified",1)
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
 
  POP_RANGE 


  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
 
  // Prefetch the data to the GPU
  char *prefetch = getenv("__PREFETCH");

  if (prefetch == NULL || strcmp(prefetch, "off") != 0) {
    std::cout << " prefetching " << std::endl ;  
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL);
  }
  else
  {
    std::cout << " not prefetching " << std::endl ;  
  }
 
  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);
 
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
 
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
