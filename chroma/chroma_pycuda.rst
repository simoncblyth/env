Getting to know how Chroma uses PyCUDA
=========================================

* http://mathema.tician.de/software/pycuda
* https://developer.nvidia.com/category/zone/cuda-zone


PyCUDA compilation in Chroma
-------------------------------

CUDA source is compiled via `pycuda.compiler.SourceModule`::

    simon:chroma blyth$ find . -name '*.py' -exec grep -H SourceModule {} \;
    ./chroma/gpu/tools.py:    """Returns a pycuda.compiler.SourceModule object from a CUDA source file
    ./chroma/gpu/tools.py:    return pycuda.compiler.SourceModule(source, options=options,
    ./chroma/gpu/tools.py:    module = pycuda.compiler.SourceModule(init_rng_src, no_extern_c=True)
    ./test/linalg_test.py:from pycuda.compiler import SourceModule
    ./test/linalg_test.py:mod = SourceModule(source, options=['-I' + source_directory], no_extern_c=True, cache_dir=False)
    ./test/matrix_test.py:from pycuda.compiler import SourceModule
    ./test/matrix_test.py:mod = SourceModule(source, options=['-I' + source_directory], no_extern_c=True, cache_dir=False)
    ./test/rotate_test.py:from pycuda.compiler import SourceModule
    ./test/rotate_test.py:mod = SourceModule(source, options=['-I' + source_directory], no_extern_c=True)
    ./test/test_sample_cdf.py:from pycuda.compiler import SourceModule
    ./test/test_sample_cdf.py:        self.mod = SourceModule(source, options=['-I' + source_directory], no_extern_c=True, cache_dir=False)
    simon:chroma blyth$ 



CUDA modules available to python
-----------------------------------

::

    simon:chroma blyth$ find . -name '*.py' -exec grep -H get_cu_module {} \;
    ./chroma/benchmark.py:    module = gpu.get_cu_module('mesh.h', options=('--use_fast_math',))

    ./chroma/camera.py:        self.gpu_funcs = gpu.GPUFuncs(gpu.get_cu_module('mesh.h'))
    ./chroma/camera.py:        self.hybrid_funcs = gpu.GPUFuncs(gpu.get_cu_module('hybrid_render.cu'))

    ./chroma/gpu/bvh.py:from chroma.gpu.tools import get_cu_module, cuda_options, \
    ./chroma/gpu/bvh.py:    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
    ./chroma/gpu/bvh.py:    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
    ./chroma/gpu/bvh.py:    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
    ./chroma/gpu/bvh.py:    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
    ./chroma/gpu/bvh.py:    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
    ./chroma/gpu/bvh.py:    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
    ./chroma/gpu/bvh.py:    bvh_module = get_cu_module('bvh.cu', options=cuda_options,

    ./chroma/gpu/daq.py:from chroma.gpu.tools import get_cu_module, cuda_options, GPUFuncs, \
    ./chroma/gpu/daq.py:        self.module = get_cu_module('daq.cu', options=cuda_options, 

    ./chroma/gpu/detector.py:from chroma.gpu.tools import get_cu_module, get_cu_source, cuda_options, \

    ./chroma/gpu/geometry.py:from chroma.gpu.tools import get_cu_module, get_cu_source, cuda_options, \
    ./chroma/gpu/geometry.py:        module = get_cu_module('mesh.h', options=cuda_options)

    ./chroma/gpu/pdf.py:from chroma.gpu.tools import get_cu_module, cuda_options, GPUFuncs, chunk_iterator
    ./chroma/gpu/pdf.py:        self.module = get_cu_module('pdf.cu', options=cuda_options,
    ./chroma/gpu/pdf.py:        self.module = get_cu_module('pdf.cu', options=cuda_options,

    ./chroma/gpu/photon.py:from chroma.gpu.tools import get_cu_module, cuda_options, GPUFuncs, \
    ./chroma/gpu/photon.py:        module = get_cu_module('propagate.cu', options=cuda_options)
    ./chroma/gpu/photon.py:        module = get_cu_module('propagate.cu', options=cuda_options)

    ./chroma/gpu/render.py:from chroma.gpu.tools import get_cu_module, cuda_options, GPUFuncs, \
    ./chroma/gpu/render.py:        transform_module = get_cu_module('transform.cu', options=cuda_options)
    ./chroma/gpu/render.py:        render_module = get_cu_module('render.cu', options=cuda_options)

    ./chroma/gpu/tools.py:def get_cu_module(name, options=None, include_source_directory=True):

    ./test/test_ray_intersection.py:        self.module = chroma.gpu.get_cu_module('mesh.h')




::

    simon:chroma blyth$ find . -name '*.cu'
    ./chroma/cuda/bvh.cu
    ./chroma/cuda/daq.cu
    ./chroma/cuda/hybrid_render.cu
    ./chroma/cuda/pdf.cu
    ./chroma/cuda/propagate.cu
    ./chroma/cuda/render.cu
    ./chroma/cuda/tools.cu
    ./chroma/cuda/transform.cu

    ./test/linalg_test.cu
    ./test/matrix_test.cu
    ./test/rotate_test.cu
    ./test/test_sample_cdf.cu
    simon:chroma blyth$ 


CUDA model of grid/blocks/threads
------------------------------------

CUDA model comprises: 

#. a grid of blocks, where each block is composed of threads
#. indexing of blocks within the grid via `blockIdx.x/y/z`  (.z always 0)

   * `gridDim.x/y/z` provides the strides indicating the number of blocks along dimensions of the grid

#. indexing of threads within the block via `threadIdx.x/y/z` 

   * threads can cooperate via shared memory 
   * `blockDim.x/y/z` provides the strides for the number of threads along dimensions of the block


Nice discussion of the model:

* http://llpanorama.wordpress.com/2008/06/11/threads-and-blocks-and-grids-oh-my/



GPU Hardware Limits
--------------------

GeForce GT 750M
~~~~~~~~~~~~~~~~~

Highlights::

      CUDA Capability Major/Minor version number:    3.0
      Total amount of global memory:                 2048 MBytes (2147024896 bytes)
      Total amount of constant memory:               65536 bytes
      Total amount of shared memory per block:       49152 bytes
      Maximum number of threads per multiprocessor:  2048
      Maximum number of threads per block:           1024
      Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
      Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
      Warp size:                                     32


#. shared memory per block only 49KB, 

   * makes sense to just share pointers into global memory, for big structures (like geometry)

#. maximum number of threads per block of 1024 seems large compared to the typical numbers
   in chroma code of 64/128/256 : maybe other limits are hit 



Full::

    Device 0: "GeForce GT 750M"
      CUDA Driver Version / Runtime Version          5.5 / 5.5
      CUDA Capability Major/Minor version number:    3.0
      Total amount of global memory:                 2048 MBytes (2147024896 bytes)
      ( 2) Multiprocessors, (192) CUDA Cores/MP:     384 CUDA Cores

      GPU Clock rate:                                926 MHz (0.93 GHz)
      Memory Clock rate:                             2508 Mhz
      Memory Bus Width:                              128-bit
      L2 Cache Size:                                 262144 bytes

      Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
      Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
      Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers

      Total amount of constant memory:               65536 bytes
      Total amount of shared memory per block:       49152 bytes
      Total number of registers available per block: 65536
      Warp size:                                     32

      Maximum number of threads per multiprocessor:  2048
      Maximum number of threads per block:           1024
      Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
      Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
      Maximum memory pitch:                          2147483647 bytes
      Texture alignment:                             512 bytes
      Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
      Run time limit on kernels:                     Yes
      Integrated GPU sharing Host Memory:            No
      Support host page-locked memory mapping:       Yes
      Alignment requirement for Surfaces:            Yes
      Device has ECC support:                        Disabled
      Device supports Unified Addressing (UVA):      Yes
      Device PCI Bus ID / PCI location ID:           1 / 0
      Compute Mode:
         < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

    deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 5.5, CUDA Runtime Version = 5.5, NumDevs = 1, Device0 = GeForce GT 750M
    Result = PASS





chroma grid/blocks
-------------------

#. Chroma uses 1D indexing, except for `chroma/cuda/bvh.cu:min_distance_to` 

   * which is used from `chroma/gpu/bvh.py:optimize_layer` which comes into play with `chroma-bvh` node_swap command


`chroma/cuda/bvh.cu`::

    445   __global__ void min_distance_to(unsigned int first_node, unsigned int threads_this_round,
    446                                   unsigned int target_index,
    447                                   uint4 *node,
    448                                   unsigned int block_offset,
    449                                   unsigned long long *min_area_block,
    450                                   unsigned int *min_index_block,
    451                                   unsigned int *flag)
    452   {
    453     __shared__ unsigned long long min_area[128];
    454     __shared__ unsigned long long adjacent_area;
    455 
    456     target_index += blockIdx.y;
    457 
    458     uint4 a = node[target_index];


::

    (chroma_env)delta:chroma blyth$ find . -name '*.py' -exec grep -H min_distance_to {} \;
    ./gpu/bvh.py:            bvh_funcs.min_distance_to(np.uint32(first_index + test_index + 2),

::

    (chroma_env)delta:chroma blyth$ find . -name '*.h' -exec grep -H Idx {} \;
    ./cuda/mesh.h:    //if (blockIdx.x == 0 && threadIdx.x == 0) {
    ./cuda/mesh.h:    if (threadIdx.x == 0)
    ./cuda/mesh.h:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/mesh.h:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/random.h:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/random.h:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/random.h:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/random.h:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    (chroma_env)delta:chroma blyth$ 


Almost always the same indexing pattern used, (bvh being the exception):: 

    (chroma_env)delta:chroma blyth$ find . -name '*.cu' -exec grep -H Idx {} \;
    ./cuda/bvh.cu:    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:    target_index += blockIdx.y;
    ./cuda/bvh.cu:    if (threadIdx.x == 0) {
    ./cuda/bvh.cu:    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:    min_area[threadIdx.x] = area;
    ./cuda/bvh.cu:    if (threadIdx.x == 0) {
    ./cuda/bvh.cu:      if (blockIdx.y == 0) {
    ./cuda/bvh.cu:          min_index_block[block_offset + blockIdx.x] = node_id;
    ./cuda/bvh.cu:          min_area_block[block_offset + blockIdx.x] = area;
    ./cuda/bvh.cu:          min_area_block[block_offset + blockIdx.x] = 0xFFFFFFFFFFFFFFFF;
    ./cuda/bvh.cu:    min_index_block[block_offset + blockIdx.x] = target_index + 1;
    ./cuda/bvh.cu:    flag[blockIdx.y] = 1;
    ./cuda/bvh.cu:     unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    ./cuda/bvh.cu:     unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    ./cuda/daq.cu:    int id = threadIdx.x + blockDim.x * blockIdx.x;
    ./cuda/daq.cu:    int id = threadIdx.x + blockDim.x * blockIdx.x;
    ./cuda/daq.cu:    if (threadIdx.x == 0) {
    ./cuda/daq.cu:  photon_id = first_photon + blockIdx.x;
    ./cuda/daq.cu:    int id = threadIdx.x + blockDim.x * blockIdx.x;
    ./cuda/daq.cu:    for (int i = threadIdx.x; i < ndaq; i += blockDim.x) {
    ./cuda/daq.cu:    int id = threadIdx.x + blockDim.x * blockIdx.x;
    ./cuda/daq.cu:    int id = threadIdx.x + blockDim.x * blockIdx.x;

    ./cuda/hybrid_render.cu:    int kernel_id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/hybrid_render.cu:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/hybrid_render.cu:    int id = blockIdx.x*blockDim.x + threadIdx.x;

    ./cuda/pdf.cu:    int id = threadIdx.x + blockDim.x * blockIdx.x;
    ./cuda/pdf.cu:    int channel_id = threadIdx.x + blockDim.x * blockIdx.x;
    ./cuda/pdf.cu:    int hit_id = threadIdx.x + blockDim.x * blockIdx.x;
    ./cuda/pdf.cu:    int hit_id = blockIdx.x;
    ./cuda/pdf.cu:    if (threadIdx.x == 0) {
    ./cuda/pdf.cu:    for (int i=threadIdx.x; i < min_bin_content; i += blockDim.x) {
    ./cuda/pdf.cu:    for (int i=threadIdx.x; i < queue_items; i += blockDim.x) {
    ./cuda/pdf.cu:    if (threadIdx.x == 0) {
    ./cuda/pdf.cu:    for (int i=threadIdx.x; i < distance_table_len; i += blockDim.x) {
    ./cuda/pdf.cu:    int id = threadIdx.x + blockDim.x * blockIdx.x;
    ./cuda/pdf.cu:    int id = threadIdx.x + blockDim.x * blockIdx.x;

    ./cuda/propagate.cu:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/propagate.cu:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/propagate.cu:    if (threadIdx.x == 0)
    ./cuda/propagate.cu:    if (threadIdx.x == 0)
    ./cuda/propagate.cu:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/propagate.cu:    if (threadIdx.x == 0)
    ./cuda/propagate.cu:    int id = blockIdx.x*blockDim.x + threadIdx.x;

    ./cuda/render.cu:    if (threadIdx.x == 0)
    ./cuda/render.cu:    int id = blockIdx.x*blockDim.x + threadIdx.x;

    ./cuda/tools.cu:    int id = blockIdx.x*blockDim.x + threadIdx.x;

    ./cuda/transform.cu:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/transform.cu:    int id = blockIdx.x*blockDim.x + threadIdx.x;
    ./cuda/transform.cu:    int id = blockIdx.x*blockDim.x + threadIdx.x;

    (chroma_env)delta:chroma blyth$ 


chroma grid
-------------

::


    (chroma_env)delta:chroma blyth$ find . -name '*.py' -exec grep -H grid= {} \;
    ./benchmark.py:        gpu_funcs.distance_to_mesh(np.int32(pos.size), pos, dir, gpu_geometry.gpudata, distances_gpu, block=(nthreads_per_block,1,1), grid=(pos.size//nthreads_per_block+1,1))
    ./camera.py:                self.hybrid_funcs.update_xyz_lookup(np.int32(self.npixels), np.int32(self.xyz_lookup1_gpu.size), np.int32(i*self.npixels), ga.vec.make_float3(*source_position), self.rng_states_gpu, np.float32(wavelength), ga.vec.make_float3(*rgb_tuple), self.xyz_lookup1_gpu, self.xyz_lookup2_gpu, np.int32(self.max_steps), self.gpu_geometry.gpudata, block=(self.nblocks,1,1), grid=(self.npixels//self.nblocks+1,1))
    ./camera.py:            self.hybrid_funcs.update_xyz_image(np.int32(rays.pos.size), self.rng_states_gpu, rays.pos, rays.dir, np.float32(wavelength), ga.vec.make_float3(*rgb_tuple), self.xyz_lookup1_gpu, self.xyz_lookup2_gpu, image, np.int32(self.nlookup_calls), np.int32(self.max_steps), self.gpu_geometry.gpudata, block=(self.nblocks,1,1), grid=(rays.pos.size//self.nblocks+1,1))
    ./camera.py:            self.hybrid_funcs.process_image(np.int32(self.pixels1_gpu.size), self.image1_gpu, self.pixels1_gpu, np.int32(self.nimages), block=(self.nblocks,1,1), grid=((self.pixels1_gpu.size)//self.nblocks+1,1))
    ./camera.py:            self.hybrid_funcs.process_image(np.int32(self.pixels2_gpu.size), self.image2_gpu, self.pixels2_gpu, np.int32(self.nimages), block=(self.nblocks,1,1), grid=((self.pixels2_gpu.size)//self.nblocks+1,1))
    ./camera.py:            self.hybrid_funcs.process_image(np.int32(self.pixels_gpu.size), self.image_gpu, self.pixels_gpu, np.int32(self.nimages), block=(self.nblocks,1,1), grid=((self.pixels_gpu.size)//self.nblocks+1,1))
    ./camera.py:                self.gpu_funcs.distance_to_mesh(np.int32(self.scope_rays.pos.size), self.scope_rays.pos, self.scope_rays.dir, gpu_geometry.gpudata, distance_gpu, block=(self.nblocks,1,1), grid=(self.scope_rays.pos.size//self.nblocks,1))
    ./gpu/bvh.py:                              grid=(nblocks_this_iter,1))
    ./gpu/bvh.py:                                        grid=(nblocks_this_iter,1))
    ./gpu/bvh.py:                                 grid=(120,1))
    ./gpu/bvh.py:                                  grid=(120,1))
    ./gpu/bvh.py:                               grid=(nblocks_this_iter,1))
    ./gpu/bvh.py:                                      grid=(nblocks_this_iter,1))
    ./gpu/bvh.py:                                    grid=(nblocks_this_iter,1))
    ./gpu/bvh.py:                                      grid=(nblocks_this_iter, skip_this_round))
    ./gpu/bvh.py:                       nodes, block=(1,1,1), grid=(1,1))
    ./gpu/bvh.py:                            grid=(nblocks_this_iter,1))
    ./gpu/daq.py:        self.gpu_funcs.reset_earliest_time_int(np.float32(1e9), np.int32(len(self.earliest_time_int_gpu)), self.earliest_time_int_gpu, block=(nthreads_per_block,1,1), grid=(len(self.earliest_time_int_gpu)//nthreads_per_block+1,1))
    ./gpu/daq.py:                                       block=(nthreads_per_block,1,1), grid=(blocks,1))
    ./gpu/daq.py:                                            block=(nthreads_per_block,1,1), grid=(blocks,1))
    ./gpu/daq.py:        self.gpu_funcs.convert_sortable_int_to_float(np.int32(len(self.earliest_time_int_gpu)), self.earliest_time_int_gpu, self.earliest_time_gpu, block=(nthreads_per_block,1,1), grid=(len(self.earliest_time_int_gpu)//nthreads_per_block+1,1))
    ./gpu/daq.py:        self.gpu_funcs.convert_charge_int_to_float(self.detector_gpu, self.channel_q_int_gpu, self.channel_q_gpu, block=(nthreads_per_block,1,1), grid=(len(self.channel_q_int_gpu)//nthreads_per_block+1,1))
    ./gpu/geometry.py:                         grid=(blocks,1))
    ./gpu/pdf.py:                                          grid=(len(gpuchannels.t)//nthreads_per_block+1,1))
    ./gpu/pdf.py:                                              grid=(len(gpuchannels.t)//nthreads_per_block+1,1))
    ./gpu/pdf.py:                                grid=(len(gpuchannels.t)//nthreads_per_block+1,1))
    ./gpu/pdf.py:                                           grid=(self.event_hit_gpu.size//nthreads_per_block+1,1))
    ./gpu/pdf.py:                                                         grid=(self.event_nhit,1))
    ./gpu/photon.py:                                                block=(nthreads_per_block,1,1), grid=(blocks, 1))
    ./gpu/photon.py:                self.gpu_funcs.propagate(np.int32(first_photon), np.int32(photons_this_round), input_queue_gpu[1:], output_queue_gpu, rng_states, self.pos, self.dir, self.wavelengths, self.pol, self.t, self.flags, self.last_hit_triangles, self.weights, np.int32(nsteps), np.int32(use_weights), np.int32(scatter_first), gpu_geometry.gpudata, block=(nthreads_per_block,1,1), grid=(blocks, 1))
    ./gpu/photon.py:                                         grid=(blocks, 1))
    ./gpu/photon.py:                                            grid=(blocks, 1))
    ./gpu/render.py:        self.transform_funcs.rotate(np.int32(self.pos.size), self.pos, np.float32(phi), ga.vec.make_float3(*n), block=(self.nblocks,1,1), grid=(self.pos.size//self.nblocks+1,1))
    ./gpu/render.py:        self.transform_funcs.rotate(np.int32(self.dir.size), self.dir, np.float32(phi), ga.vec.make_float3(*n), block=(self.nblocks,1,1), grid=(self.dir.size//self.nblocks+1,1))
    ./gpu/render.py:        self.transform_funcs.rotate_around_point(np.int32(self.pos.size), self.pos, np.float32(phi), ga.vec.make_float3(*n), ga.vec.make_float3(*point), block=(self.nblocks,1,1), grid=(self.pos.size//self.nblocks+1,1))
    ./gpu/render.py:        self.transform_funcs.rotate(np.int32(self.dir.size), self.dir, np.float32(phi), ga.vec.make_float3(*n), block=(self.nblocks,1,1), grid=(self.dir.size//self.nblocks+1,1))
    ./gpu/render.py:        self.transform_funcs.translate(np.int32(self.pos.size), self.pos, ga.vec.make_float3(*v), block=(self.nblocks,1,1), grid=(self.pos.size//self.nblocks+1,1))
    ./gpu/render.py:        self.render_funcs.render(np.int32(self.pos.size), self.pos, self.dir, gpu_geometry.gpudata, np.uint32(alpha_depth), pixels, self.dx, self.dxlen, self.color, block=(self.nblocks,1,1), grid=(self.pos.size//self.nblocks+1,1))
    ./gpu/tools.py:    init_rng(np.int32(size), rng_states, np.uint64(seed), np.uint64(0), block=(64,1,1), grid=(size//64+1,1))
    (chroma_env)delta:chroma blyth$ 




sharing between threads, sharing geometry
-------------------------------------------

* http://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/

All things shared look to be small, apart from Geometry::

    (chroma_env)delta:cuda blyth$ grep __shared__ *.cu
    bvh.cu:    __shared__ unsigned long long min_area[128];
    bvh.cu:    __shared__ unsigned long long adjacent_area;
    daq.cu:    __shared__ int photon_id;
    daq.cu:    __shared__ int triangle_id;
    daq.cu:    __shared__ int solid_id;
    daq.cu:    __shared__ int channel_index;
    daq.cu:    __shared__ unsigned int history;
    daq.cu:    __shared__ float photon_time;
    daq.cu:    __shared__ float weight;
    pdf.cu:    __shared__ float distance_table[1000];
    pdf.cu:    __shared__ unsigned int *work_queue;
    pdf.cu:    __shared__ int queue_items;
    pdf.cu:    __shared__ int channel_id;
    pdf.cu:    __shared__ float channel_event_time;
    pdf.cu:    __shared__ int distance_table_len;
    pdf.cu:    __shared__ int offset;
    propagate.cu:    __shared__ unsigned int counter;
    propagate.cu:    __shared__ Geometry sg;
    render.cu:    __shared__ Geometry sg;

    (chroma_env)delta:cuda blyth$ grep __shared__ *.h
    mesh.h:    __shared__ Geometry sg;

chroma/cuda/mesh.h::

    36 /* Finds the intersection between a ray and `geometry`. If the ray does
    37    intersect the mesh and the index of the intersected triangle is not equal
    38    to `last_hit_triangle`, set `min_distance` to the distance from `origin` to
    39    the intersection and return the index of the triangle which the ray
    40    intersected, else return -1. */
    41 __device__ int
    42 intersect_mesh(const float3 &origin, const float3& direction, Geometry *g,
    43            float &min_distance, int last_hit_triangle = -1)
    44 {
    45     int triangle_index = -1;
    46 
    47     float distance;
    48     min_distance = -1.0f;
    49 
    50     Node root = get_node(g, 0);
    ...
    ...
    ...    geometry tree traversal without using recursion  
    ...
    116 
    117     return triangle_index;
    118 }
    ...
    ...
    123 __global__ void
    124 distance_to_mesh(int nthreads, float3 *_origin, float3 *_direction,
    125          Geometry *g, float *_distance)
    126 {
    127     __shared__ Geometry sg;
    128 
    129     if (threadIdx.x == 0)
    130     sg = *g;                 //  index 0 thread copy the data from global memory to shared memory
    131 
    132     __syncthreads();         //  all threads wait until all in the block have reached here
    ...                              //  this will be repeated for thread index 0 of each block, 
    ...                              //  when there is more than one block in the grid
    133 
    134     int id = blockIdx.x*blockDim.x + threadIdx.x;
    135 
    136     if (id >= nthreads)
    137     return;
    138 
    139     g = &sg;
    140 
    141     float3 origin = _origin[id];
    142     float3 direction = _direction[id];
    143     direction /= norm(direction);
    144 
    145     float distance;
    146 
    147     int triangle_index = intersect_mesh(origin, direction, g, distance);
    148 
    149     if (triangle_index != -1)
    150     _distance[id] = distance;
    151 }


`chroma/cuda/geometry_types.h`::

     54 struct Geometry
     55 {
     56     float3 *vertices;
     57     uint3 *triangles;
     58     unsigned int *material_codes;
     59     unsigned int *colors;
     60     uint4 *primary_nodes;
     61     uint4 *extra_nodes;
     62     Material **materials;
     63     Surface **surfaces;
     64     float3 world_origin;
     65     float world_scale;
     66     int nprimary_nodes;
     67 };


`Geometry` struct is just a collection of pointers to 
large arrays of primaries, so no problem with it fitting into 49KB.
Depending on where the primaries live.

For more on geometry, :doc:`chroma_geometry`.

     

