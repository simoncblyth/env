PyCUDA Memory
==============

* device memory, host memory, pinned memory, mapped memory, free-ing memory


Observations
--------------

GPU Memory Cleanup Issue ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suspect problem with PyCUDA/Chroma GPU memory cleanup, as usually 
finding chroma propagation runtimes (observerd with non-vbo variant) 
are a factor of 3 less in the morning, at the start of work.

Freeing Device Memory
------------------------

* http://lists.tiker.net/pipermail/pycuda/2010-April/002333.html

> Can I manually free GPUarray instances?  If not, can I somehow manually
> remove all PyCUDA stuff from memory?

Python deinitialises objects as soon as the reference count for them
becomes zero. If you need to do it explicitly, I think just "del
gpuarray_obj" will be enough. At least, it worked for me.


Pagelocked (aka pinned) memory, Device mapped memory 
------------------------------------------------------ 


Chroma GPUGeometry supposedly uses device mapped memory   

`chroma/chroma/gpu/geometry.py`::

    ..1 import numpy as np
      2 import pycuda.driver as cuda
      3 from pycuda import gpuarray as ga
      4 from pycuda import characterize
      5 
      6 from collections import OrderedDict
      7 
      8 from chroma.geometry import standard_wavelengths
      9 from chroma.gpu.tools import get_cu_module, get_cu_source, cuda_options, \
     10     chunk_iterator, format_array, format_size, to_uint3, to_float3, \
     11     make_gpu_struct, GPUFuncs, mapped_empty, Mapped
     12 
     13 #from chroma.log import logger
     14 import logging
     15 log = logging.getLogger(__name__)
     16 
     17 class GPUGeometry(object):
     18     def __init__(self, geometry, wavelengths=None, print_usage=False, min_free_gpu_mem=300e6):
     19         log.info("GPUGeometry.__init__ min_free_gpu_mem %s ", min_free_gpu_mem)
     20 
    ...
    143         self.vertices = mapped_empty(shape=len(geometry.mesh.vertices),
    144                                      dtype=ga.vec.float3,
    145                                      write_combined=True)
    146         self.triangles = mapped_empty(shape=len(geometry.mesh.triangles),
    147                                       dtype=ga.vec.uint3,
    148                                       write_combined=True)
    149         self.vertices[:] = to_float3(geometry.mesh.vertices)
    150         self.triangles[:] = to_uint3(geometry.mesh.triangles)
    ...
    202         # See if there is enough memory to put the and/ortriangles back on the GPU
    203         gpu_free, gpu_total = cuda.mem_get_info()
    204         if self.triangles.nbytes < (gpu_free - min_free_gpu_mem):
    205             self.triangles = ga.to_gpu(self.triangles)
    206             log.info('Optimization: Sufficient memory to move triangles onto GPU')
    207             triangle_gpu = 1
    208         else:
    209             triangle_gpu = 0
    210         pass


pagelocked or pinned host memory 
-------------------------------------------------

* http://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/

Host (CPU) data allocations are pageable by default. The GPU cannot access data
directly from pageable host memory, so when a data transfer from pageable host
memory to device memory is invoked, the CUDA driver must first allocate a
temporary page-locked, or “pinned”, host array, copy the host data to the
pinned array, and then transfer the data from the pinned array to device
memory, as illustrated below.

As you can see in the figure, pinned memory is used as a staging area for
transfers from the device to the host. We can avoid the cost of the transfer
between pageable and pinned host arrays by directly allocating our host arrays
in pinned memory. Allocate pinned host memory in CUDA C/C++ using
cudaMallocHost() or cudaHostAlloc(), and deallocate it with cudaFreeHost().




cuMemHostAlloc : CUDA Host Memory Allocation
------------------------------------------------

TODO: find the 5.5 version of these docs

* http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDA__MEM_g572ca4011bfcb25034888a14d4e035b9.html

::

    CUresult cuMemHostAlloc (void ** pp, size_t bytesize, unsigned int Flags)   


The Flags parameter enables different options to be specified that affect the allocation, as follows.

CU_MEMHOSTALLOC_PORTABLE
      The memory returned by this call will be considered as pinned memory by all CUDA contexts, 
      not just the one that performed the allocation.

CU_MEMHOSTALLOC_DEVICEMAP
      Maps the allocation into the CUDA address space. 
      The device pointer to the memory may be obtained by calling cuMemHostGetDevicePointer(). 
      This feature is available only on GPUs with compute capability greater than or equal to 1.1.

CU_MEMHOSTALLOC_WRITECOMBINED
      Allocates the memory as write-combined (WC). 
      WC memory can be transferred across the PCI Express bus more quickly on some system configurations, 
      but cannot be read efficiently by most CPUs. 
      WC memory is a good option for buffers that will be written by the CPU and read by the GPU 
      via mapped pinned memory or host->device transfers.

All of these flags are orthogonal to one another: a developer may allocate memory 
that is portable, mapped and/or write-combined with no restrictions.

**The CUDA context must have been created with the CU_CTX_MAP_HOST flag 
in order for the CU_MEMHOSTALLOC_MAPPED flag to have any effect.**

  * is that a typo? CU_MEMHOSTALLOC_MAPPED should be CU_MEMHOSTALLOC_DEVICEMAP  
  * is Chroma/PyCUDA context being created with the requisite flag ? 


The CU_MEMHOSTALLOC_MAPPED flag may be specified on CUDA contexts for devices
that do not support mapped pinned memory. The failure is deferred to
cuMemHostGetDevicePointer() because the memory may be mapped into other CUDA
contexts via the CU_MEMHOSTALLOC_PORTABLE flag.

The memory allocated by this function must be freed with cuMemFreeHost().



cuCtxCreate
--------------

* http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDA__CTX_g65dc0012348bc84810e2103a40d8e2cf.html

::

    CUresult cuCtxCreate    (   CUcontext *     pctx, unsigned int    flags, CUdevice    dev  )   


CU_CTX_MAP_HOST
      Instruct CUDA to support mapped pinned allocations. This flag must be set in order 
      to allocate pinned host memory that is accessible to the GPU.



CU_CTX_MAP_HOST
----------------

`/usr/local/env/chroma_env/build/build_pycuda/pycuda/src/wrapper/wrap_cudadrv.cpp`::

     525 
     526 #if CUDAPP_CUDA_VERSION >= 2000
     527   py::enum_<CUctx_flags>("ctx_flags")
     528     .value("SCHED_AUTO", CU_CTX_SCHED_AUTO)
     529     .value("SCHED_SPIN", CU_CTX_SCHED_SPIN)
     530     .value("SCHED_YIELD", CU_CTX_SCHED_YIELD)
     531     .value("SCHED_MASK", CU_CTX_SCHED_MASK)
     532 #if CUDAPP_CUDA_VERSION >= 2020 && CUDAPP_CUDA_VERSION < 4000
     533     .value("BLOCKING_SYNC", CU_CTX_BLOCKING_SYNC)
     534     .value("SCHED_BLOCKING_SYNC", CU_CTX_BLOCKING_SYNC)
     535 #endif
     536 #if CUDAPP_CUDA_VERSION >= 4000
     537     .value("BLOCKING_SYNC", CU_CTX_SCHED_BLOCKING_SYNC)
     538     .value("SCHED_BLOCKING_SYNC", CU_CTX_SCHED_BLOCKING_SYNC)
     539 #endif
     540 #if CUDAPP_CUDA_VERSION >= 2020
     541     .value("MAP_HOST", CU_CTX_MAP_HOST)
     542 #endif
     543 #if CUDAPP_CUDA_VERSION >= 3020
     544     .value("LMEM_RESIZE_TO_MAX", CU_CTX_LMEM_RESIZE_TO_MAX)
     545 #endif
     546     .value("FLAGS_MASK", CU_CTX_FLAGS_MASK)
     547     ;
     548 #endif


::

    In [1]: import pycuda.driver as cuda

    In [2]: cuda.ctx_flags
    Out[2]: pycuda._driver.ctx_flags

    In [3]: cuda.ctx_flags.
    cuda.ctx_flags.BLOCKING_SYNC        cuda.ctx_flags.conjugate
    cuda.ctx_flags.FLAGS_MASK           cuda.ctx_flags.denominator
    cuda.ctx_flags.LMEM_RESIZE_TO_MAX   cuda.ctx_flags.imag
    cuda.ctx_flags.MAP_HOST             cuda.ctx_flags.mro
    cuda.ctx_flags.SCHED_AUTO           cuda.ctx_flags.name
    cuda.ctx_flags.SCHED_BLOCKING_SYNC  cuda.ctx_flags.names
    cuda.ctx_flags.SCHED_MASK           cuda.ctx_flags.numerator
    cuda.ctx_flags.SCHED_SPIN           cuda.ctx_flags.real
    cuda.ctx_flags.SCHED_YIELD          cuda.ctx_flags.values
    cuda.ctx_flags.bit_length           

    In [13]: print cuda.ctx_flags.MAP_HOST == 8
    True


chroma context creation
-------------------------

::

    (chroma_env)delta:chroma blyth$ find . -name '*.py' -exec grep -H context {} \;
    ./benchmark.py:# Generator processes need to fork BEFORE the GPU context is setup
    ./benchmark.py:    context = gpu.create_cuda_context()
    ./benchmark.py:    context.pop()
    ./camera.py:        self.context = gpu.create_cuda_context(self.device_id)
    ./camera.py:        self.context.pop()
    ./generator/photon.py:        context = zmq.Context()
    ./generator/photon.py:        vertex_socket = context.socket(zmq.PULL)
    ./generator/photon.py:        photon_socket = context.socket(zmq.PUSH)
    ./generator/photon.py:        self.zmq_context = zmq.Context()
    ./generator/photon.py:        self.vertex_socket = self.zmq_context.socket(zmq.PUSH)
    ./generator/photon.py:        self.photon_socket = self.zmq_context.socket(zmq.PULL)
    ./gpu/tools.py:    Hashability needed for context_dependent_memoize dictates the type
    ./gpu/tools.py:@pycuda.tools.context_dependent_memoize
    ./gpu/tools.py:def create_cuda_context(device_id=None):
    ./gpu/tools.py:    """Initialize and return a CUDA context on the specified device.
    ./gpu/tools.py:            context = pycuda.tools.make_default_context()
    ./gpu/tools.py:            context = pycuda.tools.make_default_context()
    ./gpu/tools.py:        context = device.make_context()
    ./gpu/tools.py:    context.set_cache_config(cuda.func_cache.PREFER_L1)
    ./gpu/tools.py:    return context
    ./loader.py:from chroma.gpu import create_cuda_context
    ./loader.py:        context = create_cuda_context(cuda_device)
    ./loader.py:        context.pop()
    ./sim.py:        self.context = gpu.create_cuda_context(cuda_device)
    ./sim.py:        self.context.pop()
    (chroma_env)delta:chroma blyth$ 



g4daechroma/g4daeview DAEChromaContext 
----------------------------------------

`env/geant4/geometry/collada/g4daeview/daechromacontext.py`::

     16 import numpy as np
     17 import pycuda.gl.autoinit  # after this can use pycuda.gl.BufferObject(unsigned int)
     18 
     19 def pick_seed():
     20     """Returns a seed for a random number generator selected using
     21     a mixture of the current time and the current process ID."""
     22     return int(time.time()) ^ (os.getpid() << 16)
     23 
     24 class DAEChromaContext(object):



::

    In [3]: from pycuda.gl import autoinit

    In [4]: autoinit??
    Type:       module
    String Form:<module 'pycuda.gl.autoinit' from '/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/gl/autoinit.pyc'>
    File:       /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/gl/autoinit.py
    Source:
    import pycuda.driver as cuda
    import pycuda.gl as cudagl

    cuda.init()
    assert cuda.Device.count() >= 1

    from pycuda.tools import make_default_context
    context = make_default_context(lambda dev: cudagl.make_context(dev))
    device = context.get_device()

    import atexit
    atexit.register(context.pop)






pycuda context creation
--------------------------

`/usr/local/env/chroma_env/build/build_pycuda/pycuda/pycuda/tools.py`::

    159 def make_default_context(ctx_maker=None):
    160     if ctx_maker is None:
    161         def ctx_maker(dev):
    162             return dev.make_context()
    163 
    164     ndevices = cuda.Device.count()
    ...
    194     # Otherwise, try to use any available device
    195     else:
    196         for devn in xrange(ndevices):
    197             dev = cuda.Device(devn)
    198             try:
    199                 return ctx_maker(dev)
    200             except cuda.Error:
    201                 pass


`/usr/local/env/chroma_env/build/build_pycuda/pycuda/src/wrapper/wrap_cudadrv.cpp`::

     867   // {{{ device
     868   {
     869     typedef device cl;
     870     py::class_<cl>("Device", py::no_init)
     871       .def("__init__", py::make_constructor(make_device))
     872 #if CUDAPP_CUDA_VERSION >= 4010
     873       .def("__init__", py::make_constructor(make_device_from_pci_bus_id))
     874 #endif
     875       .DEF_SIMPLE_METHOD(count)
     876       .staticmethod("count")
     877       .DEF_SIMPLE_METHOD(name)
     878 #if CUDAPP_CUDA_VERSION >= 4010
     879       .DEF_SIMPLE_METHOD(pci_bus_id)
     880 #endif
     881       .DEF_SIMPLE_METHOD(compute_capability)
     882       .DEF_SIMPLE_METHOD(total_memory)
     883       .def("get_attribute", device_get_attribute)
     884       .def(py::self == py::self)
     885       .def(py::self != py::self)
     886       .def("__hash__", &cl::hash)
     887       .def("make_context", &cl::make_context,
     888           (py::args("self"), py::args("flags")=0))
     889 #if CUDAPP_CUDA_VERSION >= 4000
     890       .DEF_SIMPLE_METHOD(can_access_peer)
     891 #endif
     892       ;
     893   }

`/usr/local/env/chroma_env/build/build_pycuda/pycuda/src/cpp/cuda.hpp`::

     766 
     767   inline
     768   boost::shared_ptr<context> device::make_context(unsigned int flags)
     769   {
     770     context::prepare_context_switch();
     771 
     772     CUcontext ctx;
     773     CUDAPP_CALL_GUARDED(cuCtxCreate, (&ctx, flags, m_device));
     774     boost::shared_ptr<context> result(new context(ctx));
     775     context_stack::get().push(result);
     776     return result;
     777   }
     778 
     779 




mapped_empty 
-------------

The largest Chroma arrays (bvh nodes, vertices, triangles) are 
all handled using **mapped_empty**. These use allocator 
**pycuda.driver.pagelocked_empty** with mem flags:

* pycuda.driver.host_alloc_flags.DEVICEMAP (CU_MEMHOSTALLOC_DEVICEMAP)
* pycuda.driver.host_alloc_flags.WRITECOMBINED (CU_MEMHOSTALLOC_WRITECOMBINED)


`chroma/chroma/gpu/tools.py`::

    ..8 import pycuda.driver as cuda
    ...
    247 def mapped_alloc(pagelocked_alloc_func, shape, dtype, write_combined):
    248     '''Returns a pagelocked host array mapped into the CUDA device
    249     address space, with a gpudata field set so it just works with CUDA 
    250     functions.'''
    251     flags = cuda.host_alloc_flags.DEVICEMAP
    252     if write_combined:
    253         flags |= cuda.host_alloc_flags.WRITECOMBINED
    254     array = pagelocked_alloc_func(shape=shape, dtype=dtype, mem_flags=flags)
    255     return array
    256 
    257 def mapped_empty(shape, dtype, write_combined=False):
    258     '''See mapped_alloc()'''
    259     return mapped_alloc(cuda.pagelocked_empty, shape, dtype, write_combined)


`/usr/local/env/chroma_env/build/build_pycuda/pycuda/src/wrapper/wrap_cudadrv.cpp`::

     ..1 #include <cuda.hpp>
     ...
     .79   class host_alloc_flags { };
     ...
     810 #if CUDAPP_CUDA_VERSION >= 2020
     811   {
     812     py::class_<host_alloc_flags> cls("host_alloc_flags", py::no_init);
     813     cls.attr("PORTABLE") = CU_MEMHOSTALLOC_PORTABLE;
     814     cls.attr("DEVICEMAP") = CU_MEMHOSTALLOC_DEVICEMAP;
     815     cls.attr("WRITECOMBINED") = CU_MEMHOSTALLOC_WRITECOMBINED;
     816   }
     817 #endif
     818 
     819 #if CUDAPP_CUDA_VERSION >= 4000
     820   {
     821     py::class_<mem_host_register_flags> cls("mem_host_register_flags", py::no_init);
     822     cls.attr("PORTABLE") = CU_MEMHOSTREGISTER_PORTABLE;
     823     cls.attr("DEVICEMAP") = CU_MEMHOSTREGISTER_DEVICEMAP;
     824   }
     825 #endif


Memory Pools
--------------

* http://documen.tician.de/pycuda/util.html

The functions pycuda.driver.mem_alloc() and pycuda.driver.pagelocked_empty()
can consume a fairly large amount of processing time if they are invoked very
frequently. For example, code based on pycuda.gpuarray.GPUArray can easily run
into this issue because a fresh memory area is allocated for each intermediate
result. Memory pools are a remedy for this problem based on the observation
that often many of the block allocations are of the same sizes as previously
used ones.

Then, instead of fully returning the memory to the system and incurring the
associated reallocation overhead, the pool holds on to the memory and uses it
to satisfy future allocations of similarly-sized blocks. The pool reacts
appropriately to out-of-memory conditions as long as all memory allocations are
made through it. Allocations performed from outside of the pool may run into
spurious out-of-memory conditions due to the pool owning much or all of the
available memory.




ga.to_gpu
-----------

* creates GPUArray instance which 

  * allocates device memory 
  * copies htod with  `drv.memcpy_htod(self.gpudata, ary)`


`/usr/local/env/chroma_env/build/build_pycuda/pycuda/pycuda/gpuarray.py`::

     861 # {{{ creation helpers
     862 
     863 def to_gpu(ary, allocator=drv.mem_alloc):
     864     """converts a numpy array to a GPUArray"""
     865     result = GPUArray(ary.shape, ary.dtype, allocator, strides=ary.strides)
     866     result.set(ary)
     867     return result
     ...
     137 class GPUArray(object):
     138     """A GPUArray is used to do array-based calculation on the GPU.
     139 
     140     This is mostly supposed to be a numpy-workalike. Operators
     141     work on an element-by-element basis, just like numpy.ndarray.
     142     """
     143 
     144     __array_priority__ = 100
     145 
     146     def __init__(self, shape, dtype, allocator=drv.mem_alloc,
     147             base=None, gpudata=None, strides=None, order="C"):
     148         dtype = np.dtype(dtype)
     149 
     ...
     184         self.allocator = allocator
     185         if gpudata is None:
     186             if self.size:
     187                 self.gpudata = self.allocator(self.size * self.dtype.itemsize)
     188             else:
     189                 self.gpudata = None
     190 
     191             assert base is None
     192         else:
     193             self.gpudata = gpudata
     194 
     195         self.base = base
     196 
     197         self._grid, self._block = splay(self.mem_size)
     ...
     204     def set(self, ary):
     205         assert ary.size == self.size
     206         assert ary.dtype == self.dtype
     207         if ary.strides != self.strides:
     208             from warnings import warn
     209             warn("Setting array from one with different strides/storage order. "
     210                     "This will cease to work in 2013.x.",
     211                     stacklevel=2)
     212 
     213         assert self.flags.forc
     214 
     215         if self.size:
     216             drv.memcpy_htod(self.gpudata, ary)
     217 



to_float3 to_uint3
----------------------

* copies (N,3) into float32
* subclasses to ga.vec.float3
* reshape to yield (N)


`chroma/chroma/gpu/tools.py`::

    137 def to_float3(arr):
    138     "Returns an pycuda.gpuarray.vec.float3 array from an (N,3) array."
    139     if not arr.flags['C_CONTIGUOUS']:
    140         arr = np.asarray(arr, order='c')
    141     return arr.astype(np.float32).view(ga.vec.float3)[:,0]
    142 
    143 def to_uint3(arr):
    144     "Returns a pycuda.gpuarray.vec.uint3 array from an (N,3) array."
    145     if not arr.flags['C_CONTIGUOUS']:
    146         arr = np.asarray(arr, order='c')
    147     return arr.astype(np.uint32).view(ga.vec.uint3)[:,0]


::

    In [1]: a = np.arange(10)

    In [2]: a.flags
    Out[2]: 
      C_CONTIGUOUS : True
      F_CONTIGUOUS : True
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      UPDATEIFCOPY : False

    a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)

    Copy of the array, cast to a specified type.

    a.view(dtype=None, type=None)

    New view of array with the same data.

    Parameters
    ----------
    dtype : data-type or ndarray sub-class, optional
        Data-type descriptor of the returned view, e.g., float32 or int16. The
        default, None, results in the view having the same data-type as `a`.
        This argument can also be specified as an ndarray sub-class, which
        then specifies the type of the returned object (this is equivalent to
        setting the ``type`` parameter).
    type : Python type, optional
        Type of the returned view, e.g., ndarray or matrix.  Again, the
        default None results in type preservation.

    In [8]: a = np.arange(30).reshape(10,3)

    In [9]: a
    Out[9]: 
    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11],
           [12, 13, 14],
           [15, 16, 17],
           [18, 19, 20],
           [21, 22, 23],
           [24, 25, 26],
           [27, 28, 29]])

    In [10]: a[:,0]
    Out[10]: array([ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27])

    In [11]: a[:,0].shape
    Out[11]: (10,)





