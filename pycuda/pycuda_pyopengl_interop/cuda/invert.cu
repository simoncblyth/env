extern "C" 
{

__global__ void invert(unsigned char *source, unsigned char *dest)
{
  int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
  int thread_num       = threadIdx.y * blockDim.x + threadIdx.x;
  int threads_in_block = blockDim.x * blockDim.y;
  //Since the image is RGBA we multiply the index 4.
  //We'll only use the first 3 (RGB) channels though
  int idx              = 4 * (threads_in_block * block_num + thread_num);
  dest[idx  ] = 255 - source[idx  ];
  dest[idx+1] = 255 - source[idx+1];
  dest[idx+2] = 255 - source[idx+2];
}


__global__ void generate(unsigned char *dest)
{
  int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
  int thread_num       = threadIdx.y * blockDim.x + threadIdx.x;
  int threads_in_block = blockDim.x * blockDim.y;
  int idx              = 4 * (threads_in_block * block_num + thread_num);

  dest[idx  ] = 255 - idx % 255 ;
  dest[idx+1] = idx % 255 ; 
  dest[idx+2] = idx % 255 ; 
}




} // extern "C"

