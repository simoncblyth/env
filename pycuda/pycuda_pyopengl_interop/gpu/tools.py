#!/usr/bin/env python
"""
Extracted tools from chroma/gpu/tools.py that are generally 
useful and not chroma specific at all

* http://documen.tician.de/pycuda/driver.html#pycuda.compiler.SourceModule

Note the use of `no_extern_c=True` in `get_cu_module` means will need
have that in the cuda source.::

   extern "C" {

   __global__ void kernelname1(... )
   {
   ...
   }

   __global__ void kernelname2(... )
   {
   ...
   }


   }  // extern C
   

"""
import numpy as np
import pytools
import pycuda.tools
from pycuda import characterize
import pycuda.driver as cuda
import pycuda.compiler

from env.pycuda.pycuda_pyopengl_interop.cuda import srcdir

# standard nvcc options
cuda_options = ('--use_fast_math',)#, '--ptxas-options=-v']

@pycuda.tools.context_dependent_memoize
def get_cu_module(name, options=None, include_source_directory=True):
    """Returns a pycuda.compiler.SourceModule object from a CUDA source file
    located in the chroma cuda directory at cuda/[name]."""
    if options is None:
        options = []
    elif isinstance(options, tuple):
        options = list(options)
    else:
        raise TypeError('`options` must be a tuple.')

    if include_source_directory:
        options += ['-I' + srcdir]

    with open('%s/%s' % (srcdir, name)) as f:
        source = f.read()

    return pycuda.compiler.SourceModule(source, options=options, no_extern_c=True)

@pytools.memoize
def get_cu_source(name):
    """Get the source code for a CUDA source file located in the chroma cuda
    directory at src/[name]."""
    with open('%s/%s' % (srcdir, name)) as f:
        source = f.read()
    return source

class GPUFuncs(object):
    """Simple container class for GPU functions as attributes."""
    def __init__(self, module):
        self.module = module
        self.funcs = {}

    def __getattr__(self, name):
        try:
            return self.funcs[name]
        except KeyError:
            f = self.module.get_function(name)
            self.funcs[name] = f
            return f

def chunk_iterator(nelements, nthreads_per_block=64, max_blocks=1024):
    """Iterator that yields tuples with the values requried to process
    a long array in multiple kernel passes on the GPU.

    Each yielded value is of the form,
        (first_index, elements_this_iteration, nblocks_this_iteration)

    Example:
        >>> list(chunk_iterator(300, 32, 2))
        [(0, 64, 2), (64, 64, 2), (128, 64, 2), (192, 64, 2), (256, 9, 1)]
    """
    first = 0
    while first < nelements:
        elements_left = nelements - first
        blocks = int(elements_left // nthreads_per_block)
        if elements_left % nthreads_per_block != 0:
            blocks += 1 # Round up only if needed
        blocks = min(max_blocks, blocks)
        elements_this_round = min(elements_left, blocks * nthreads_per_block)

        yield (first, elements_this_round, blocks)
        first += elements_this_round

def create_cuda_context(device_id=None):
    """Initialize and return a CUDA context on the specified device.
    If device_id is None, the default device is used."""
    if device_id is None:
        try:
            context = pycuda.tools.make_default_context()
        except cuda.LogicError:
            # initialize cuda
            cuda.init()
            context = pycuda.tools.make_default_context()
    else:
        try:
            device = cuda.Device(device_id)
        except cuda.LogicError:
            # initialize cuda
            cuda.init()
            device = cuda.Device(device_id)
        context = device.make_context()

    context.set_cache_config(cuda.func_cache.PREFER_L1)

    return context

def Mapped(array):
    '''Analog to pycuda.driver.InOut(), but indicates this array
    is memory mapped to the device space and should not be copied.

    To simplify coding, Mapped() will pass anything with a gpudata
    member, like a gpuarray, through unchanged.
    '''
    if hasattr(array, 'gpudata'):
        return array
    else:
        return np.intp(array.base.get_device_pointer())


