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




