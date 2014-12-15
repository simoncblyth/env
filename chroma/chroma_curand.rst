Chroma Use Of CURand
=====================

* https://devtalk.nvidia.com/default/topic/485547/investigating-rngs-for-large-numbers-of-parallel-streams-/


* https://bitbucket.org/chroma/chroma/src/b565b38ae23a5b7522b54af51091e2f7c4267c9c/chroma/gpu/tools.py?at=default



init_rng
---------

#. hmm small fixed threads_per_block of 64 for `init_rng` likely 
   cause of slow `init_rng`  




`chroma/chroma/gpu/tools.py`::

    107 init_rng_src = """
    108 #include <curand_kernel.h>
    109 
    110 extern "C"
    111 {
    112 
    113 __global__ void init_rng(int nthreads, curandState *s, unsigned long long seed, unsigned long long offset)
    114 {
    115     int id = blockIdx.x*blockDim.x + threadIdx.x;
    116 
    117     if (id >= nthreads)
    118         return;
    119 
    120     curand_init(seed, id, offset, &s[id]);
    121 }
    122 
    123 } // extern "C"
    124 """
    125 
    126 def get_rng_states(size, seed=1):
    127     "Return `size` number of CUDA random number generator states."
    128     rng_states = cuda.mem_alloc(size*characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))
    129 
    130     module = pycuda.compiler.SourceModule(init_rng_src, no_extern_c=True)
    131     init_rng = module.get_function('init_rng')
    132 
    133     init_rng(np.int32(size), rng_states, np.uint64(seed), np.uint64(0), block=(64,1,1), grid=(size//64+1,1))
    134 
    135     return rng_states





