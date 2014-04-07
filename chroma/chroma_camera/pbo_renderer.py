#!/usr/bin/env python
"""
"""
import logging
log = logging.getLogger(__name__)

import numpy as np

from env.cuda.cuda_launch import Launch

import pycuda.gl as cuda_gl
import pycuda.driver as cuda
import pycuda.gpuarray as ga
import pycuda.gl.autoinit      # excludes use of non-gl autoinit

from chroma.gpu.geometry import GPUGeometry
from chroma.gpu.tools import get_cu_module, cuda_options


class PBORenderer(object):
    def __init__(self, pixels, chroma_geometry, config ):
        pass
        self.pixels = pixels
        self.config = config
        self.gpu_geometry = GPUGeometry( chroma_geometry )

        self.compile_kernel()
        self.launch = Launch( pixels.npixels, threads_per_block=config.args.threads_per_block, max_blocks=config.args.max_blocks )

        self._size = None
        self._origin = None
        self._pixel2world = None

        if hasattr(config, 'eye'): 
            self.size = pixels.size
            self.origin = config.eye
            self.pixel2world = config.pixel2world

    def compile_kernel(self):
        """
        #. compile kernel and extract __constant__ symbol addresses
        """
        module = get_cu_module('render_pbo.cu', options=cuda_options)
        self.arg_format = "iiPP"

        self.g_size   = module.get_global("g_size")[0]  
        self.g_origin = module.get_global("g_origin")[0]
        self.g_pixel2world = module.get_global("g_pixel2world")[0]  

        kernel = module.get_function(self.config.args.kernel)
        kernel.prepare(self.arg_format)
        self.kernel = kernel

    def _get_size(self):
        return self._size 
    def _set_size(self, size):
        self._size = size
        cuda.memcpy_htod(self.g_size,         ga.vec.make_int2(*size))
    size = property(_get_size, _set_size) 
      
    def _get_origin(self):
        return self._origin
    def _set_origin(self, origin):
        self._origin = origin
        cuda.memcpy_htod(self.g_origin,       ga.vec.make_float4(*origin))
    origin = property(_get_origin, _set_origin) 
 
    def _get_pixel2world(self):
        return self._pixel2world
    def _set_pixel2world(self, pixel2world):
        self._pixel2world = pixel2world
        cuda.memcpy_htod(self.g_pixel2world,  np.float32(pixel2world))
    pixel2world = property(_get_pixel2world, _set_pixel2world) 
 
    def render(self, alpha_depth=3):
        """
        :param alpha_depth:

        * http://stackoverflow.com/questions/6954487/how-to-use-the-prepare-function-from-pycuda

        """
        assert alpha_depth <= self.config.args.max_alpha_depth

        pbo_mapping = self.pixels.cuda_pbo.map()

        args = [ np.uint32(alpha_depth), 
                 pbo_mapping.device_ptr(),
                 self.gpu_geometry.gpudata ]

        log.info("render pixels %s launch %s " % (repr(self.pixels.size), repr(self.launch)))

        block = self.launch.block
        calls = 0 
        times = []
        for offset, count, blocks_per_grid in self.launch.chunker:

            grid=(blocks_per_grid, 1)
            get_time = self.kernel.prepared_timed_call( grid, block, np.uint32(offset), *args )
            t = get_time()
            times.append(t)
 
            log.info("[%s] offset %s grid %s took %s " % (calls, offset, repr(grid), t ))
            if self.config.args.allsync:
                cuda.Context.synchronize()  
            pass
            calls += 1
        pass
        cuda.Context.synchronize()  # OMITTING THIS SYNC CAN CAUSE AN IRRECOVERABLE GUI FREEZE
        pbo_mapping.unmap()



